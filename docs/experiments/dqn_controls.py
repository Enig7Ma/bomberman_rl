"""Play the control policies over a fixed board set, 60 rounds in all.

Three policies x two scenarios x ten paired maps. Inference only: no training
imports, no new heuristic, no mutation of source artifacts. Run from the
repository root::

    uv run python -m docs.experiments.dqn_controls --out results/dqn/controls
"""

import argparse
import hashlib
import json
import random
import subprocess
from collections.abc import Callable
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np

from agent_code.dqn_agent import callbacks as dqn
from agent_code.dqn_agent.encoder import ENCODERS
from agent_code.dqn_agent.network import QNetwork
from agent_code.tabular_q_agent import callbacks as tabular
from agent_code.tabular_q_agent.features import ENCODINGS
from agent_code.tabular_q_agent.qtable import QTable
from docs.experiments.dqn_d6 import checksum, dump
from tournament import engine
from tournament.engine import WorldConfig, quiet_logging
from training.driver import environment
from training.world import TrainingWorld, create_training_world

MODEL = Path("agent_code/tabular_q_agent/model/q_table.npz")
DQN_ROOT = Path("results/dqn/d7_stage2_seed0_20260918")
NETWORK = DQN_ROOT / "checkpoints/transition_000350346/q_net.npz"
SOURCES = {
    "d6_parent": DQN_ROOT / "checkpoints/transition_000050047/evaluation",
    "dqn_lr3e4": DQN_ROOT / "checkpoints/transition_000350346/evaluation",
    "dqn_lr1e4": Path(
        "results/dqn/d7_stage2_lr1e4_seed0_20260918/checkpoints/transition_000350443/evaluation"
    ),
}


def backtracks(positions: list[tuple[int, int]]) -> int:
    return sum(
        positions[i] == positions[i - 2] != positions[i - 1]
        for i in range(2, len(positions))
    )


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    result: dict[str, Any] = {
        "rounds": n,
        "per_map_coins": [r["coins"] for r in rows],
        "seeds": [r["seed"] for r in rows],
    }
    for key in (
        "coins",
        "crates",
        "bombs",
        "suicides",
        "deaths",
        "steps",
        "invalid",
        "timeouts",
    ):
        total = sum(r[key] for r in rows)
        result[key] = {"total": total, "mean_per_round": total / n}
    for key in ("wait", "backtracks"):
        result[key] = (
            None if any(r.get(key) is None for r in rows) else sum(r[key] for r in rows)
        )
    return result


def run(directory: Path) -> None:
    if directory.exists():
        raise FileExistsError("fresh output required; no automatic repeated games")
    shared = (
        "features.py",
        "mask.py",
        "symmetry.py",
        "core/world_model.py",
        "core/safety.py",
        "core/planning.py",
        "core/attack.py",
        "core/params.py",
    )
    for name in shared:
        assert (Path("agent_code/dqn_agent") / name).read_bytes() == (
            Path("agent_code/tabular_q_agent") / name
        ).read_bytes()
    table = QTable.load(MODEL, ENCODINGS["E3"])
    assert table.visited_states > 0 and table.meta["steps_trained"] > 0
    net = QNetwork.load(NETWORK, ENCODERS["onehot_e3"])
    protected = [MODEL]
    for root in (
        DQN_ROOT,
        Path("results/dqn/d7_stage2_lr1e4_seed0_20260918"),
        Path("results/dqn/d6_20260917/pilot_lr3e-04_target1000/run_0"),
    ):
        protected.extend(
            p
            for p in root.rglob("*")
            if p.is_file() and p.suffix in (".npz", ".pt", ".json", ".jsonl")
        )
    hashes = {str(p): checksum(p) for p in protected}
    directory.mkdir(parents=True)
    dump(
        directory / "protocol.json",
        {
            "cap": 60,
            "world_seeds": list(range(500, 510)),
            "private_policy_seed": 500,
            "train": False,
            "features": "E3",
            "mask": "best_tier",
            "table": str(MODEL),
            "table_meta": table.meta,
            "table_commit": subprocess.check_output(
                ["git", "log", "-1", "--format=%H", "--", str(MODEL)], text=True
            ).strip(),
            "code_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "script_sha256": checksum(Path(__file__)),
            "before_sha256": hashes,
            "table_limitation": (
                "artifact tracked with training metadata; full curriculum history "
                "unavailable; reference only"
            ),
        },
    )
    boards: dict[tuple[str, int], str] = {}
    rows: list[dict[str, Any]] = []
    started_games = 0
    for policy in ("random", "heuristic", "table"):
        module = dqn if policy == "random" else tabular
        original = cast(Callable[..., str], module.act)
        agent_name = "dqn_agent" if policy == "random" else "tabular_q_agent"
        prefix = "DQN_AGENT" if policy == "random" else "TABULAR_Q_AGENT"
        model = NETWORK if policy == "random" else MODEL
        params = {
            "seed": 500,
            "mask": "best_tier",
            "policy": "learned" if policy == "table" else policy,
        }
        params["encoder" if policy == "random" else "encoding"] = (
            "onehot_e3" if policy == "random" else "E3"
        )
        for scenario in ("loot-crate", "classic"):
            for seed in range(500, 510):
                positions: list[tuple[int, int]] = []
                actions: list[str] = []

                @wraps(original)
                def capture(
                    agent: object,
                    state: dict[str, Any],
                    positions: list[tuple[int, int]] = positions,
                    actions: list[str] = actions,
                    original: Callable[..., str] = original,
                    scenario: str = scenario,
                    seed: int = seed,
                ) -> str:
                    if not positions:
                        board = hashlib.sha256(
                            np.asarray(state["field"]).tobytes()
                            + repr((state["self"][3], state["coins"])).encode()
                        ).hexdigest()
                        key = (scenario, seed)
                        assert boards.setdefault(key, board) == board
                    x, y = state["self"][3]
                    positions.append((int(x), int(y)))
                    action = str(original(agent, state))
                    actions.append(action)
                    return action

                def world_factory(
                    config: WorldConfig,
                    log_dir: str = "logs",
                    policy: str = policy,
                    model: Path = model,
                ) -> TrainingWorld:
                    world = create_training_world(
                        config.lineup,
                        config.scenario,
                        config.seed,
                        train_seats=0,
                        log_dir=log_dir,
                    )
                    handle = cast(Any, world).agents[0].backend.runner.fake_self
                    assert not handle.train and handle.trainer is None
                    assert isinstance(handle.rng, random.Random)
                    assert handle.rng.getstate() == random.Random(500).getstate()
                    assert (
                        handle.config.mask == "best_tier"
                        and handle.encoding.name == "E3"
                    )
                    assert handle.config.policy == (
                        "learned" if policy == "table" else policy
                    )
                    assert handle.model_file.resolve() == model.resolve()
                    if policy == "random":
                        assert isinstance(handle.q_function, QNetwork)
                        for key, value in net.arrays.items():
                            np.testing.assert_array_equal(
                                handle.q_function.arrays[key], value
                            )
                    else:
                        np.testing.assert_array_equal(handle.table.q, table.q)
                        np.testing.assert_array_equal(handle.table.n, table.n)
                    return world

                started_games += 1
                assert started_games <= 60
                dump(
                    directory / "started_games.json",
                    {
                        "count": started_games,
                        "policy": policy,
                        "scenario": scenario,
                        "seed": seed,
                    },
                )
                with (
                    environment(
                        {
                            f"{prefix}_MODEL": str(model.resolve()),
                            f"{prefix}_PARAMS": json.dumps(params),
                        }
                    ),
                    quiet_logging(),
                    patch.object(module, "act", capture),
                    patch.object(engine, "create_world", world_factory),
                ):
                    result = engine.run_round(
                        WorldConfig((agent_name,), scenario, seed),
                        log_dir=str(directory / "logs" / f"{policy}_{scenario}_{seed}"),
                    )
                focus = result.focus
                row = {
                    "policy": policy,
                    "scenario": scenario,
                    "seed": seed,
                    **{
                        k: getattr(focus, k)
                        for k in (
                            "coins",
                            "crates",
                            "bombs",
                            "suicides",
                            "steps",
                            "invalid",
                            "timeouts",
                        )
                    },
                    "deaths": int(not focus.survived),
                    "wait": actions.count("WAIT"),
                    "backtracks": backtracks(positions),
                }
                rows.append(row)
                dump(
                    directory / f"{policy}_{scenario}_{seed}.json",
                    {
                        "result": asdict(result),
                        "actions": actions,
                        "positions": positions,
                    },
                )
                dump(directory / "new_rows.json", rows)
                print(json.dumps(row), flush=True)
    for name, source in SOURCES.items():
        for scenario, preset in (
            ("loot-crate", "crates-solo"),
            ("classic", "classic-solo"),
        ):
            old = [
                json.loads(x)
                for x in (source / f"{preset}.jsonl").read_text().splitlines()
            ]
            traces = json.loads((source / f"{preset}_decisions.json").read_text())
            assert [r["seed"] for r in old] == list(range(500, 510))
            for item, trace in zip(old, traces, strict=True):
                focus = item["agents"][0]
                assert item["scenario"] == scenario and item["focus_seat"] == 0
                rows.append(
                    {
                        "policy": name,
                        "scenario": scenario,
                        "seed": item["seed"],
                        **{
                            k: focus[k]
                            for k in (
                                "coins",
                                "crates",
                                "bombs",
                                "suicides",
                                "steps",
                                "invalid",
                                "timeouts",
                            )
                        },
                        "deaths": int(not focus["survived"]),
                        "wait": sum(x["action"] == "WAIT" for x in trace),
                        "backtracks": None,
                    }
                )
    summaries = [
        {
            "policy": policy,
            "scenario": scenario,
            **aggregate(
                [r for r in rows if r["policy"] == policy and r["scenario"] == scenario]
            ),
        }
        for policy in ("random", "heuristic", "table", *SOURCES)
        for scenario in ("loot-crate", "classic")
    ]
    assert hashes == {str(p): checksum(p) for p in protected}
    dump(
        directory / "summary.json",
        {
            "new_games": started_games,
            "unchanged_files": len(hashes),
            "rows": rows,
            "summaries": summaries,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.out)


if __name__ == "__main__":
    main()
