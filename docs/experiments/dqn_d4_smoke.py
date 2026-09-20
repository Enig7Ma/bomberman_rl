"""Bounded D4 stock-framework smoke (50 rounds), not a curriculum driver."""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from agent_code.dqn_agent.config import ENV_VAR, METRICS_ENV_VAR, MODEL_ENV_VAR, Config
from agent_code.dqn_agent.encoder import OneHotE3
from agent_code.dqn_agent.learner import Learner
from agent_code.dqn_agent.metrics import read_records
from agent_code.dqn_agent.network import QNetwork
from agent_code.dqn_agent.persistence import (
    load_checkpoint,
    load_replay,
    save_checkpoint,
    save_replay,
)
from agent_code.dqn_agent.replay import ReplayBuffer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    config = Config(seed=0, init_seed=0)
    initial = Learner(OneHotE3(), config).export()
    initial.save(output / "initial_untrained.npz")
    model = output / "q_net.npz"
    metrics = output / "metrics.jsonl"
    env = {
        **os.environ,
        ENV_VAR: json.dumps(asdict(config)),
        MODEL_ENV_VAR: str(model),
        METRICS_ENV_VAR: str(metrics),
    }
    command = [
        sys.executable,
        "main.py",
        "play",
        "--no-gui",
        "--agents",
        "dqn_agent",
        "--train",
        "1",
        "--scenario",
        "coin-heaven",
        "--n-rounds",
        "50",
        "--seed",
        "1700",
        "--log-dir",
        str(output / "logs"),
    ]
    started = perf_counter()
    with (output / "framework_output.txt").open("w", encoding="utf-8") as log:
        subprocess.run(
            command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    elapsed = perf_counter() - started
    records = read_records(metrics)
    checkpoint = load_checkpoint(output / "checkpoint.pt")
    replay, degraded = load_replay(
        output / "replay.npz", checkpoint, OneHotE3().schema_id
    )
    final = QNetwork.load(model, OneHotE3())
    assert len(records) == 50 and not degraded
    assert records[-1]["updates"] > 0
    assert all(r["updates"] == 0 for r in records if r["transitions"] < config.warmup)
    changed = any(
        not np.array_equal(v, final.arrays[k]) for k, v in initial.arrays.items()
    )
    assert changed
    assert checkpoint["transitions"] == records[-1]["transitions"] == len(replay)
    assert (
        checkpoint["learner"]["updates"]
        == final.meta["updates"]
        == records[-1]["updates"]
    )
    assert records[0]["resume_mode"] == "random-init"
    timings: dict[str, list[float]] = {
        "q_net": [],
        "checkpoint": [],
        "replay_filled": [],
        "replay_100k": [],
    }
    # Capacity measurement: repeat actual smoke rows, with fresh unique IDs.
    # This measures 100k storage, not a claim of 100k trained transitions.
    state = replay.snapshot()
    rows = np.resize(state["rows"], 100_000)
    rows["transition_id"] = np.arange(100_000, dtype=np.uint64)
    full = ReplayBuffer.from_snapshot(
        {
            "capacity": 100_000,
            "input_dim": 32,
            "size": 100_000,
            "write": 0,
            "rows": rows,
        }
    )
    costs = output / "save_costs"
    costs.mkdir()
    for _ in range(5):
        start = perf_counter()
        final.save(costs / "q_net.npz")
        timings["q_net"].append(perf_counter() - start)
        timings["checkpoint"].append(
            save_checkpoint(costs / "checkpoint.pt", checkpoint)
        )
        for name, buffer, count in (
            ("replay_filled", replay, checkpoint["transitions"]),
            ("replay_100k", full, 100_000),
        ):
            timings[name].append(
                save_replay(
                    costs / f"{name}.npz",
                    buffer,
                    run_id=checkpoint["run_id"],
                    transitions=count,
                    schema_id=OneHotE3().schema_id,
                )
            )
    summary: dict[str, Any] = {
        "command": command,
        "config": asdict(config),
        "rounds": len(records),
        "transitions": records[-1]["transitions"],
        "updates": records[-1]["updates"],
        "weights_changed": changed,
        "exact_replay_checkpoint": not degraded,
        "first_round_with_updates": next(r["round"] for r in records if r["updates"]),
        "seconds": elapsed,
        "last_metrics": records[-1],
        "replay_100k_raw_bytes": rows.nbytes,
        "save_seconds": timings,
        "save_mean_ms": {k: float(np.mean(v) * 1000) for k, v in timings.items()},
        "bytes": {p.name: p.stat().st_size for p in costs.iterdir()},
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
