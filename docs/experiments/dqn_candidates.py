"""Bounded practical validation; no training or replacement of shipped weights."""

import argparse
import hashlib
import importlib
import json
import random
import subprocess
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from time import perf_counter
from typing import Any, cast
from unittest.mock import patch

import numpy as np

from agent_code.dqn_agent.encoder import OneHotE3
from agent_code.dqn_agent.network import QNetwork
from agent_code.tabular_q_agent.features import ENCODINGS
from agent_code.tabular_q_agent.qtable import QTable
from docs.experiments.dqn_d6 import checksum, dump
from tournament import engine
from tournament.engine import WorldConfig, quiet_logging
from training.driver import environment
from training.world import TrainingWorld, create_training_world

DQN = Path("results/dqn/d7_stage2_seed0_20260918/checkpoints/transition_000350346")
TABLE = Path("agent_code/tabular_q_agent/model/q_table.npz")


@contextmanager
def rule_streams(seed: int, reference: bool) -> Generator[dict[int, int]]:
    """Private RNG per seat; restore globals even when stock setup seeds entropy."""
    module = importlib.import_module("agent_code.rule_based_agent.callbacks")
    original_setup, original_act = module.setup, module.act
    seats: dict[int, int] = {}
    streams: dict[int, tuple[np.random.RandomState, random.Random]] = {}

    def setup(agent: object) -> None:
        seat = len(seats) + (0 if reference else 1)
        seats[id(agent)] = seat
        streams[id(agent)] = (
            np.random.RandomState(seed * 10 + seat),
            random.Random(seed * 10 + seat),
        )
        ns, ps = np.random.get_state(), random.getstate()
        try:
            original_setup(agent)
        finally:
            np.random.set_state(ns)
            random.setstate(ps)

    def act(agent: object, state: dict[str, Any]) -> str:
        ns, ps = np.random.get_state(), random.getstate()
        nr, pr = streams[id(agent)]
        np.random.set_state(nr.get_state())
        random.setstate(pr.getstate())
        try:
            return str(original_act(agent, state))
        finally:
            nr.set_state(np.random.get_state())
            pr.setstate(random.getstate())
            np.random.set_state(ns)
            random.setstate(ps)

    with patch.object(module, "setup", setup), patch.object(module, "act", act):
        yield seats


def run(
    out: Path,
    agents_to_check: tuple[str, ...] = (
        "dqn_agent",
        "tabular_q_agent",
        "rule_based_agent",
    ),
    reference_results: Path | None = None,
) -> None:
    out = out.resolve()
    if out.exists():
        raise FileExistsError("fresh directory required; do not repeat existing games")
    net = QNetwork.load(DQN / "q_net.npz", OneHotE3())
    table = QTable.load(TABLE, ENCODINGS["E3"])
    assert table.visited_states > 0
    protected = [TABLE, *DQN.glob("*.pt"), *DQN.glob("*.npz")]
    hashes = {str(p): checksum(p) for p in protected}
    out.mkdir(parents=True)
    dump(
        out / "protocol.json",
        {
            "seeds": list(range(500, 525)),
            "scenario": "classic",
            "seat": 0,
            "max_steps": 400,
            "games": 25 * len(agents_to_check),
            "jobs": 1,
            "train": False,
            "candidate_rng": 500,
            "rule_rng": "10 * world_seed + seat",
            "exploration": 0,
            "protected": hashes,
            "head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "script_sha256": checksum(Path(__file__)),
        },
    )
    boards: dict[int, str] = {}
    if reference_results is not None:
        previous = json.loads(reference_results.read_text())
        boards = {
            row["seed"]: row["initial_board_sha256"]
            for row in previous["rows"]["rule_based_agent"]
        }
        assert set(boards) == set(range(500, 525))
    results: dict[str, list[dict[str, Any]]] = {}
    started = perf_counter()
    for name in agents_to_check:
        rows: list[dict[str, Any]] = []
        reference = name == "rule_based_agent"
        prefix = "DQN_AGENT" if name == "dqn_agent" else "TABULAR_Q_AGENT"
        model = (DQN / "q_net.npz" if name == "dqn_agent" else TABLE).resolve()
        module = importlib.import_module(f"agent_code.{name}.callbacks")
        for seed in range(500, 525):
            handle: list[Any] = []
            first: list[str] = []

            def world(
                config: WorldConfig,
                log_dir: str = "logs",
                handle: list[Any] = handle,
                reference: bool = reference,
                model: Path = model,
                name: str = name,
            ) -> TrainingWorld:
                value = create_training_world(
                    config.lineup,
                    config.scenario,
                    config.seed,
                    train_seats=0,
                    log_dir=log_dir,
                )
                agents = cast(Any, value).agents
                assert all(not a.train for a in agents)
                h = agents[0].backend.runner.fake_self
                handle[:] = [h]
                if not reference:
                    assert h.trainer is None and h.model_file.resolve() == model
                    assert h.config.policy == "learned" and h.config.mask == "best_tier"
                    assert h.rng.getstate() == random.Random(500).getstate()
                    if name == "dqn_agent":
                        assert isinstance(h.q_function, QNetwork)
                        for key in net.arrays:
                            np.testing.assert_array_equal(
                                h.q_function.arrays[key], net.arrays[key]
                            )
                    else:
                        np.testing.assert_array_equal(h.table.q, table.q)
                        np.testing.assert_array_equal(h.table.n, table.n)
                return value

            with rule_streams(seed, reference):
                original_act = module.act

                @wraps(original_act)
                def act(
                    agent: object,
                    state: dict[str, Any],
                    handle: list[Any] = handle,
                    first: list[str] = first,
                    seed: int = seed,
                    original_act: Callable[..., str] = original_act,
                ) -> str:
                    if agent is handle[0] and not first:
                        signature = hashlib.sha256(
                            np.asarray(state["field"]).tobytes()
                            + repr(
                                (
                                    state["self"][3],
                                    state["coins"],
                                    [o[3] for o in state["others"]],
                                )
                            ).encode()
                        ).hexdigest()
                        assert boards.setdefault(seed, signature) == signature
                        first.append(signature)
                    return str(original_act(agent, state))

                with (
                    environment(
                        {
                            f"{prefix}_MODEL": str(model),
                            f"{prefix}_PARAMS": json.dumps(
                                {"seed": 500, "policy": "learned"}
                            ),
                        }
                    ),
                    quiet_logging(),
                    patch.object(module, "act", act),
                    patch.object(engine, "create_world", world),
                ):
                    result = engine.run_round(
                        WorldConfig(
                            (name, *("rule_based_agent",) * 3), "classic", seed
                        ),
                        log_dir=str(out / "logs" / f"{name}_{seed}"),
                    )
            assert first
            rows.append(
                {"seed": seed, "initial_board_sha256": first[0], **asdict(result.focus)}
            )
            dump(out / f"{name}.json", rows)
        results[name] = rows
        print(name, "completed", flush=True)
    assert hashes == {str(p): checksum(p) for p in protected}
    dump(
        out / "results.json",
        {
            "rows": results,
            "seconds": perf_counter() - started,
            "weights_unchanged": True,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--agent", choices=("dqn_agent", "tabular_q_agent", "rule_based_agent")
    )
    parser.add_argument("--reference-results", type=Path)
    args = parser.parse_args()
    selected = (
        (args.agent,)
        if args.agent
        else ("dqn_agent", "tabular_q_agent", "rule_based_agent")
    )
    run(args.out, selected, args.reference_results)
