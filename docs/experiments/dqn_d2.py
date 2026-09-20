"""D2 only: evaluate one untrained NumPy network, sequentially (jobs=1).

Run from the repository root with ``python -m docs.experiments.dqn_d2
--output results/dqn/d2_random_20260916``. Refuses to overwrite an existing run.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np

from agent_code.dqn_agent.callbacks import AgentSelf
from agent_code.dqn_agent.config import ENV_VAR, MODEL_ENV_VAR, Config
from agent_code.dqn_agent.encoder import OneHotE3
from agent_code.dqn_agent.network import QNetwork
from tournament.engine import (
    collect_result,
    create_world,
    quiet_logging,
    reset_framework_logging,
)
from tournament.latency import LatencyRecord, instrument_world
from tournament.schedule import PRESETS, build_schedule


def evaluate(output: Path, name: str, seed_set: range, model: Path) -> dict[str, Any]:
    schedule = build_schedule(
        "dqn_agent", PRESETS["vs-rule-based"], seed_set, with_control=False
    )
    samples = LatencyRecord()
    errors: list[str] = []
    loaded = suicides = all_timeouts = invalid = 0
    with (output / f"{name}.jsonl").open("w", encoding="utf-8") as file:
        with quiet_logging():
            for number, config in enumerate(schedule, 1):
                try:
                    world = create_world(config, str(output / "logs"))
                    backend: Any = world.agents[config.focus_seat]
                    agent = cast(AgentSelf, backend.backend.runner.fake_self)
                    net = agent.q_function
                    assert not agent.train and agent.config.policy == "learned"
                    assert agent.model_file == model and isinstance(net, QNetwork)
                    assert net.meta["stage"] == "D2-untrained"
                    loaded += 1
                    records = instrument_world(world)
                    world.new_round()
                    while world.running:
                        world.do_step()
                    assert agent.q_function is net and agent.config.policy == "learned"
                    assert "torch" not in sys.modules
                    result = collect_result(world, config, records)
                    focus = result.agents[config.focus_seat]
                    raw = records[config.focus_seat].samples
                    assert raw, "no decisions recorded"
                    samples.samples.extend(raw)
                    samples.timeouts += focus.timeouts
                    all_timeouts += sum(a.timeouts for a in result.agents)
                    suicides += focus.suicides
                    invalid += focus.invalid
                    row = {
                        "result": asdict(result),
                        "network_loaded": True,
                        "fallback": False,
                        "latency_seconds": raw,
                    }
                except Exception:
                    error = traceback.format_exc()
                    errors.append(error)
                    row = {"config": asdict(config), "error": error}
                finally:
                    reset_framework_logging()
                file.write(json.dumps(row) + "\n")
                file.flush()
                if number % 10 == 0:
                    print(
                        f"{name}: {number}/{len(schedule)}, errors={len(errors)}",
                        flush=True,
                    )
    return {
        "scheduled_rounds": len(schedule),
        "completed_rounds": len(schedule) - len(errors),
        "loaded_network_rounds": loaded,
        "errors": errors,
        "suicides": suicides,
        "suicide_rate": suicides / len(schedule),
        "invalid_actions": invalid,
        "timeouts": samples.timeouts,
        "all_agents_timeouts": all_timeouts,
        "decisions": len(samples.samples),
        "mean_ms": samples.mean * 1000,
        "p99_ms": samples.percentile(99) * 1000,
        "max_ms": samples.maximum * 1000,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnose-model", type=Path)
    parser.add_argument("--diagnose-seed", type=int, default=506)
    args = parser.parse_args()
    output = cast(Path, args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.diagnose_model is not None:
        from .dqn_d2_trace import diagnose

        diagnose(output, Path(args.diagnose_model).resolve(), int(args.diagnose_seed))
        return
    config = Config(seed=0, init_seed=0, policy="learned")
    net = QNetwork.random(OneHotE3(), config.init_seed)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    net.meta.update(
        {"config": asdict(config), "stage": "D2-untrained", "git_commit": commit}
    )
    model = output / "random_init_q_net.npz"
    net.save(model)
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    os.environ[MODEL_ENV_VAR] = str(model)
    os.environ[ENV_VAR] = json.dumps(asdict(config))
    acceptance = evaluate(output, "acceptance", range(500, 525), model)
    latency = evaluate(output, "latency", range(300, 310), model)
    assert hashlib.sha256(model.read_bytes()).hexdigest() == digest
    summary = {
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "jobs": 1,
        "preset": "vs-rule-based",
        "acceptance_seeds": [500, 524],
        "latency_seeds": [300, 309],
        "seat_rotations": 4,
        "config": asdict(config),
        "worktree_base_commit": commit,
        "model_sha256": digest,
        "header": net.header,
        "meta": net.meta,
        "acceptance": acceptance,
        "latency": latency,
        "criteria": {
            "100_rounds_without_errors": acceptance["completed_rounds"] == 100,
            "suicide_rate_below_0.10": acceptance["suicide_rate"] < 0.10,
            "latency_mean_below_2_ms": latency["mean_ms"] < 2,
            "latency_max_below_50_ms": latency["max_ms"] < 50,
            "latency_rounds_without_errors": latency["completed_rounds"] == 40,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
