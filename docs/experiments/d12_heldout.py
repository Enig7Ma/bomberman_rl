"""Execute the frozen D12 schedule; persist each game and refuse failed retries."""

import argparse
import hashlib
import importlib
import itertools
import json
import os
import random
import sys
import time
import traceback
from collections.abc import Callable, Generator, Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from functools import update_wrapper
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np

from tournament import engine
from tournament.latency import instrument_world
from tournament.schedule import PRESETS, lineup_with
from training.world import create_training_world

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "docs/experiments/d12_frozen_protocol.json"
STOCK = {"rule_based_agent", "peaceful_agent", "coin_collector_agent"}
type Streams = dict[int, tuple[np.random.RandomState, random.Random]]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def schedule(protocol: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates = protocol["candidates"]
    seeds = protocol["seeds"]
    if mode == "smoke":
        seeds = [500]
        candidates = candidates[:2]
    if mode == "latency":
        seeds = seeds[:25]
        candidates = candidates[:2]
    for candidate in [*candidates, *([] if mode == "latency" else [None])]:
        for name, definition in protocol["presets"].items():
            if mode == "latency" and name != "vs-rule-based":
                continue
            preset = PRESETS[name]
            for seed in seeds:
                for seat in [0] if mode == "latency" else definition["seats"]:
                    identifier = candidate["id"] if candidate else "reference"
                    agent = candidate["agent"] if candidate else "rule_based_agent"
                    lineup = lineup_with(agent, preset.opponents, seat)
                    rows.append(
                        {
                            "id": f"{identifier}__{name}__{seed}__{seat}",
                            "candidate": identifier,
                            "preset": name,
                            "seed": seed,
                            "seat": seat,
                            "lineup": lineup,
                            "scenario": preset.scenario,
                            "models": {agent: identifier} if candidate else {},
                            "identities": {},
                        }
                    )
    if mode != "latency":
        hseeds = [500] if mode == "smoke" else protocol["head_to_head"]["seeds"]
        for seed in hseeds:
            for number, labels in enumerate(
                itertools.permutations(
                    ("dqn_agent", "tabular_q_agent", "rule_A", "rule_B")
                )
            ):
                rows.append(
                    {
                        "id": f"head_to_head__{seed}__{number}",
                        "candidate": "head_to_head",
                        "preset": "head_to_head",
                        "seed": seed,
                        "seat": labels.index("dqn_agent"),
                        "lineup": tuple(
                            "rule_based_agent" if x.startswith("rule_") else x
                            for x in labels
                        ),
                        "scenario": "classic",
                        "models": {
                            "dqn_agent": "dqn_stage2_baseline",
                            "tabular_q_agent": "tabular_blended",
                        },
                        "identities": {
                            str(i): int(x == "rule_B")
                            for i, x in enumerate(labels)
                            if x.startswith("rule_")
                        },
                    }
                )
    return rows


@contextmanager
def opponent_streams(
    lineup: tuple[str, ...], seed: int, identities: dict[str, int]
) -> Generator[None]:
    """Seat-local NumPy/Python streams, including entropy-seeding stock setup."""
    with ExitStack() as stack:
        for name in set(lineup) & STOCK:
            module = importlib.import_module(f"agent_code.{name}.callbacks")
            seats = iter([i for i, item in enumerate(lineup) if item == name])
            streams: dict[int, tuple[np.random.RandomState, random.Random]] = {}

            def setup(
                agent: object,
                original: Callable[..., object] = module.setup,
                seats: Iterator[int] = seats,
                streams: Streams = streams,
            ) -> None:
                seat = next(seats)
                value = 10 * seed + seat + 1_000_000 * identities.get(str(seat), 0)
                streams[id(agent)] = (
                    np.random.RandomState(value),
                    random.Random(value),
                )
                ns, ps = np.random.get_state(), random.getstate()
                try:
                    original(agent)
                finally:
                    np.random.set_state(ns)
                    random.setstate(ps)

            def act(
                agent: object,
                state: dict[str, Any],
                original: Callable[..., str] = module.act,
                streams: Streams = streams,
            ) -> str:
                ns, ps = np.random.get_state(), random.getstate()
                nr, pr = streams[id(agent)]
                np.random.set_state(nr.get_state())
                random.setstate(pr.getstate())
                try:
                    return str(original(agent, state))
                finally:
                    nr.set_state(np.random.get_state())
                    pr.setstate(random.getstate())
                    np.random.set_state(ns)
                    random.setstate(ps)

            update_wrapper(setup, module.setup)
            update_wrapper(act, module.act)
            stack.enter_context(patch.object(module, "setup", setup))
            stack.enter_context(patch.object(module, "act", act))
        yield


def play(task: dict[str, Any], models: Path, logs: Path) -> dict[str, Any]:
    lineup = tuple(task["lineup"])
    variables: dict[str, str] = {}
    expected: dict[str, dict[str, np.ndarray[Any, Any]]] = {}
    for agent, identifier in task["models"].items():
        prefix = "DQN_AGENT" if agent == "dqn_agent" else "TABULAR_Q_AGENT"
        path = (models / f"{identifier}.npz").resolve()
        variables[f"{prefix}_MODEL"] = str(path)
        variables[f"{prefix}_PARAMS"] = json.dumps({"seed": 500, "policy": "learned"})
        with np.load(path, allow_pickle=False) as saved:
            keys = (
                ("W0", "b0", "W1", "b1", "W2", "b2")
                if agent == "dqn_agent"
                else ("q", "n")
            )
            expected[agent] = {key: saved[key].copy() for key in keys}
    clean = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("DQN_AGENT_", "TABULAR_Q_AGENT_"))
    }
    config = engine.WorldConfig(
        lineup, task["scenario"], task["seed"], focus_seat=task["seat"]
    )
    started = time.perf_counter()
    try:
        with (
            patch.dict(os.environ, {**clean, **variables}, clear=True),
            opponent_streams(lineup, task["seed"], task["identities"]),
            engine.quiet_logging(),
        ):
            world = create_training_world(
                lineup, config.scenario, config.seed, train_seats=0, log_dir=str(logs)
            )
            raw = cast(Any, world)
            agents = raw.agents
            for a in agents:
                assert not a.train
                if a.code_name in expected:
                    h = a.backend.runner.fake_self
                    assert h.trainer is None and h.config.policy == "learned"
                    assert h.config.mask == "best_tier"
                    assert h.model_file == Path(
                        variables[
                            (
                                "DQN_AGENT"
                                if a.code_name == "dqn_agent"
                                else "TABULAR_Q_AGENT"
                            )
                            + "_MODEL"
                        ]
                    )
                    assert h.rng.getstate() == random.Random(500).getstate()
                    actual = (
                        h.q_function.arrays
                        if a.code_name == "dqn_agent"
                        else {"q": h.table.q, "n": h.table.n}
                    )
                    for key, array in expected[a.code_name].items():
                        np.testing.assert_array_equal(actual[key], array)
            records = instrument_world(world)
            world.new_round()
            board = hashlib.sha256(
                np.asarray(raw.arena).tobytes()
                + repr(
                    (
                        [(c.x, c.y) for c in raw.coins],
                        [(a.x, a.y) for a in agents],
                    )
                ).encode()
            ).hexdigest()
            while world.running:
                world.do_step()
            result = engine.collect_result(world, config, records)
            assert "torch" not in sys.modules
            return {
                **task,
                "initial_board_sha256": board,
                "result": asdict(result),
                "seconds": time.perf_counter() - started,
                "latencies": [r.samples for r in records],
                "status": "complete",
            }
    finally:
        engine.reset_framework_logging()


def worker(arguments: tuple[dict[str, Any], str, str]) -> str:
    task, model_dir, output = arguments
    out = Path(output)
    target = out / "games" / f"{task['id']}.json"
    if target.exists():
        old = json.loads(target.read_text())
        if old["status"] != "complete":
            raise RuntimeError(f"Retained failed game requires review: {target}")
        return task["id"]
    try:
        row = play(task, Path(model_dir), out / "logs" / str(os.getpid()))
    except Exception:
        write(target, {**task, "status": "failed", "error": traceback.format_exc()})
        raise
    write(target, row)
    return task["id"]


def run(out: Path, models: Path, mode: str) -> None:
    protocol = json.loads(PROTOCOL.read_text())
    tasks = schedule(protocol, mode)
    if mode == "heldout":
        assert len(tasks) == protocol["budget_games"]["total"]
    hashes = {c["id"]: c["sha256"] for c in protocol["candidates"]}
    for identifier, digest in hashes.items():
        assert sha(models / f"{identifier}.npz") == digest
    # Inference modules must be exactly those already package-tested in Linux.
    for package in protocol["packages"].values():
        for name, digest in package["members"].items():
            if name.endswith(".py"):
                assert sha(ROOT / "agent_code" / name) == digest
    marker: dict[str, Any] = {
        "protocol_sha256": sha(PROTOCOL),
        "script_sha256": sha(Path(__file__)),
        "models": hashes,
        "mode": mode,
        "games": len(tasks),
        "jobs": 1 if mode == "latency" else 2,
    }
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run.json").exists():
        assert json.loads((out / "run.json").read_text()) == marker
    else:
        write(out / "run.json", marker)
    start = time.perf_counter()
    arguments = [(t, str(models), str(out)) for t in tasks]
    with ProcessPoolExecutor(max_workers=marker["jobs"]) as pool:
        for count, _ in enumerate(pool.map(worker, arguments, chunksize=1), 1):
            if count % 50 == 0 or count == len(tasks):
                write(
                    out / "progress.json",
                    {
                        "complete": count,
                        "total": len(tasks),
                        "session_seconds": time.perf_counter() - start,
                    },
                )
                print(f"{count}/{len(tasks)}", flush=True)
    for identifier, digest in hashes.items():
        assert sha(models / f"{identifier}.npz") == digest
    write(out / "complete.json", {**marker, "seconds": time.perf_counter() - start})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("smoke", "heldout", "latency"), required=True
    )
    args = parser.parse_args()
    run(args.out.resolve(), args.models.resolve(), args.mode)
