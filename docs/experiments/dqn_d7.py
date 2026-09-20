"""Continue a navigation run into the crate stage from its full saved state.

Drives the curriculum in ``CONFIG``, measures the probe at each chunk boundary
and evaluates the archived weights through explicit NumPy inference.
``--smoke`` shortens the run to a plumbing check. Run from the repository
root::

    uv run python -m docs.experiments.dqn_d7 --out results/dqn/crate_stage
    uv run python -m docs.experiments.dqn_d7 --out results/dqn/d7_smoke --smoke
"""

import argparse
import json
import shutil
import time
import traceback
from collections.abc import Sequence
from dataclasses import replace
from functools import wraps
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np

from agent_code.dqn_agent import callbacks
from agent_code.dqn_agent.core.world_model import ACTIONS, Observation
from agent_code.dqn_agent.features import Extracted, Extractor
from agent_code.dqn_agent.metrics import read_records
from agent_code.dqn_agent.network import QNetwork
from agent_code.dqn_agent.persistence import load_checkpoint
from agent_code.dqn_agent.symmetry import canonical, to_canonical
from agent_code.dqn_agent.train import Trainer
from docs.experiments.dqn_d6 import (
    ENCODER,
    VALIDATION_SEEDS,
    checksum,
    dump,
    version_info,
)
from tournament import engine
from tournament.engine import WorldConfig
from tournament.runner import run_schedule
from tournament.schedule import PRESETS, build_schedule
from tournament.storage import write_rounds
from training import dqn
from training.config import load_curriculum
from training.continuation import fork_curriculum
from training.driver import environment, train_run
from training.spec import AgentSpec
from training.world import TrainingWorld, create_training_world

PROBE_SHA256 = "008253a6aa52af1e006b0d3858591f4a0cc94c03729073e659fb7ce010e3189f"
PARENT = Path("results/dqn/d6_20260917/pilot_lr3e-04_target1000/run_0")
CONFIG = Path("docs/experiments/dqn_d7_stage2.json")


def discounted_returns(
    scores: list[float], potentials: list[float], final_score: float, gamma: float
) -> list[float]:
    """Realized shaped returns, terminal potential zero; includes failed rounds."""
    result = [0.0] * len(scores)
    future = 0.0
    for index in reversed(range(len(scores))):
        terminal = index == len(scores) - 1
        next_score = final_score if terminal else scores[index + 1]
        next_phi = 0.0 if terminal else potentials[index + 1]
        reward = next_score - scores[index] + gamma * next_phi - potentials[index]
        future = reward + gamma * future
        result[index] = future
    return result


def evaluate(model: Path, directory: Path) -> list[dict[str, Any]]:
    """Thirty solo rounds, NumPy only, on the parent run's validation maps."""
    expected = QNetwork.load(model, ENCODER)
    original = callbacks.setup
    original_act, original_extract = callbacks.act, Extractor.extract
    traces: list[list[dict[str, Any]]] = []
    extracted_now: list[Extracted] = []

    @wraps(original)
    def checked_setup(agent: object) -> None:
        handle = cast(Any, agent)
        original(handle)
        traces.append([])
        assert not handle.train and handle.trainer is None
        assert isinstance(handle.q_function, QNetwork)
        assert handle.model_file.resolve() == model.resolve()
        for key in expected.arrays:
            np.testing.assert_array_equal(
                handle.q_function.arrays[key], expected.arrays[key]
            )

    def capture_extract(extractor: Extractor, obs: Observation) -> Extracted:
        extracted = original_extract(extractor, obs)
        extracted_now[:] = [extracted]
        return extracted

    @wraps(original_act)
    def capture_act(agent: object, game_state: dict[str, Any]) -> str:
        handle = cast(Any, agent)
        action = original_act(handle, game_state)
        extracted = extracted_now[0]
        index, symmetry = canonical(extracted.features, handle.encoding)
        x = ENCODER.encode(handle.encoding.decode(index), extracted)
        allowed = [ACTIONS.index(to_canonical(a, symmetry)) for a in extracted.allowed]
        distance = extracted.coin_distance
        traces[-1].append(
            {
                "score": float(game_state["self"][1]),
                "phi": 0.0 if distance is None else 0.5 / (1 + distance),
                "max_q": float(expected.values(x)[allowed].max()),
                "action": action,
                "unproductive_context": extracted.features.bomb_yield == 0
                and extracted.features.attack == 0,
            }
        )
        return action

    def evaluation_world(config: WorldConfig, log_dir: str = "logs") -> TrainingWorld:
        return create_training_world(
            config.lineup, config.scenario, config.seed, train_seats=0, log_dir=log_dir
        )

    rows: list[dict[str, Any]] = []
    for preset in ("crates-solo", "classic-solo", "coin-heaven-solo"):
        traces.clear()
        schedule = build_schedule(
            "dqn_agent", PRESETS[preset], VALIDATION_SEEDS, with_control=False
        )
        with (
            environment(
                {
                    "DQN_AGENT_MODEL": str(model.resolve()),
                    "DQN_AGENT_PARAMS": json.dumps(
                        {"seed": 500, "policy": "learned", "encoder": "onehot_e3"}
                    ),
                }
            ),
            patch.object(callbacks, "setup", checked_setup),
            patch.object(callbacks, "act", capture_act),
            patch.object(Extractor, "extract", capture_extract),
            patch.object(engine, "create_world", evaluation_world),
        ):
            results = run_schedule(
                schedule, jobs=1, log_dir=str(directory / "logs" / preset)
            )
        write_rounds(directory / f"{preset}.jsonl", results)
        gaps: list[float] = []
        for trace, result in zip(traces, results, strict=True):
            returns = discounted_returns(
                [row["score"] for row in trace],
                [row["phi"] for row in trace],
                float(result.focus.score),
                0.99,
            )
            for row, value in zip(trace, returns, strict=True):
                row["discounted_shaped_return"] = value
                gaps.append(row["max_q"] - value)
        dump(directory / f"{preset}_decisions.json", traces)
        n = len(results)
        total_steps = sum(r.focus.steps for r in results)
        totals = {
            key: sum(getattr(r.focus, key) for r in results)
            for key in (
                "coins",
                "crates",
                "suicides",
                "bombs",
                "invalid",
                "score",
                "timeouts",
            )
        }
        rows.append(
            {
                "preset": preset,
                "rounds": n,
                "agent_steps": total_steps,
                "mean_q_minus_realized_return": float(np.mean(gaps)),
                "mean_abs_q_return_gap": float(np.mean(np.abs(gaps))),
                "unproductive_contexts": sum(
                    row["unproductive_context"] for trace in traces for row in trace
                ),
                "bombs_in_unproductive_context": sum(
                    row["unproductive_context"] and row["action"] == "BOMB"
                    for trace in traces
                    for row in trace
                ),
                "totals": totals,
                "per_round": {key: value / n for key, value in totals.items()},
                "deaths": sum(not r.focus.survived for r in results),
                "mean_steps": sum(r.steps for r in results) / n,
                "steps": [r.steps for r in results],
                "all50_rounds": sum(r.focus.coins == 50 for r in results),
                "bombs_per_action": totals["bombs"] / total_steps,
                "invalid_per_action": totals["invalid"] / total_steps,
                "quality_pass": totals["coins"] / n > 43.05 and totals["suicides"] == 0
                if preset == "crates-solo"
                else None,
            }
        )
    dump(directory / "summary.json", rows)
    return rows


def run(directory: Path, *, smoke: bool = False) -> None:
    """One branch only; failed runs are never resumed automatically."""
    course = load_curriculum(CONFIG)
    probe = Path(course.params["probe_path"])
    assert checksum(probe) == PROBE_SHA256
    if smoke:
        course = replace(
            course,
            chunk_rounds=1,
            stages=(replace(course.stages[0], transitions=600, rounds=2),),
        )
    parent_hashes = {
        name: checksum(PARENT / name)
        for name in ("checkpoint.pt", "replay.npz", "q_net.npz")
    }
    trainer = fork_curriculum(PARENT, directory, course, 0)
    origin, origin_updates = trainer.transitions, trainer.learner.updates
    active = [trainer]
    dump(directory / "environment.json", version_info())
    shutil.copyfile(Path(__file__), directory / "experiment_source.py")
    elapsed_evaluation = 0.0
    snapshots: list[dict[str, Any]] = []

    def archive(current: Trainer) -> None:
        nonlocal elapsed_evaluation
        assert current.probe is not None
        diagnostic = current.probe.measure(current.learner.online)
        current.save(full=True)
        path = directory / "checkpoints" / f"transition_{current.transitions:09d}"
        path.mkdir(parents=True, exist_ok=True)
        for name in ("checkpoint.pt", "replay.npz", "q_net.npz"):
            shutil.copyfile(directory / name, path / name)
        dump(path / "probe.json", diagnostic)
        started = time.perf_counter()
        hashes = {name: checksum(directory / name) for name in parent_hashes}
        evaluation = [] if smoke else evaluate(path / "q_net.npz", path / "evaluation")
        assert hashes == {name: checksum(directory / name) for name in parent_hashes}
        elapsed_evaluation += time.perf_counter() - started
        snapshots.append(
            {
                "global_transitions": current.transitions,
                "stage_transitions": current.transitions - origin,
                "updates": current.learner.updates,
                "checkpoint": str(path / "checkpoint.pt"),
                "checkpoint_sha256": checksum(path / "checkpoint.pt"),
                "probe": diagnostic,
                "evaluation": evaluation,
            }
        )
        dump(directory / "snapshots_summary.json", snapshots)
        print(json.dumps(snapshots[-1]), flush=True)

    original_save = AgentSpec.save_chunk
    original_world = dqn.create_training_world

    def world_hook(
        lineup: Sequence[str],
        scenario: str,
        seed: int,
        *,
        train_seats: int = 1,
        log_dir: str = "logs",
    ) -> TrainingWorld:
        world = original_world(
            lineup, scenario, seed, train_seats=train_seats, log_dir=log_dir
        )
        active[:] = [cast(Any, world).agents[0].backend.runner.fake_self.trainer]
        return world

    def save_hook(spec: AgentSpec, agent: object) -> None:
        current = cast(Trainer, cast(Any, agent).trainer)
        active[:] = [current]
        assert current.driver_state is not None
        records = current.driver_state["records"]
        if records and records[-1]["snapshot"]:
            archive(current)
        else:
            original_save(spec, agent)

    started = time.perf_counter()
    failure = None
    try:
        archive(trainer)
        with (
            patch.object(AgentSpec, "save_chunk", save_hook),
            patch.object(dqn, "create_training_world", world_hook),
        ):
            train_run(course, directory, 0)
        if smoke:
            before = load_checkpoint(directory / "checkpoint.pt")
            assert train_run(course, directory, 0) == []
            after = load_checkpoint(directory / "checkpoint.pt")
            assert before["transitions"] == after["transitions"]
    except Exception:
        failure = traceback.format_exc()
        (directory / "failure.txt").write_text(failure, encoding="utf-8")
        dump(
            directory / "stop.json",
            {
                "transitions": active[0].transitions,
                "stage_position": active[0].stage_position,
                "updates": active[0].learner.updates,
                "pending": active[0].bookkeeper.pending is not None,
            },
        )
        stopped = active[0]
        if stopped.probe is not None:
            np.savez_compressed(
                directory / "stopped_probe_q.npz",
                q=np.stack([stopped.learner.online.values(x) for x in stopped.probe.x]),
            )
        # No automatic retry after a diagnostic stop.
    seconds = time.perf_counter() - started - elapsed_evaluation
    committed = load_checkpoint(directory / "checkpoint.pt")
    metrics_path = directory / "metrics.jsonl"
    metrics = read_records(metrics_path) if metrics_path.exists() else []
    dump(
        directory / "summary.json",
        {
            "failure": failure,
            "smoke": smoke,
            "parent_sha256": parent_hashes,
            "probe_sha256": checksum(probe),
            "training_seconds": seconds,
            "evaluation_seconds": elapsed_evaluation,
            "global_transitions": committed["transitions"],
            "stage_transitions": committed["transitions"] - origin,
            "global_updates": committed["learner"]["updates"],
            "stage_updates": committed["learner"]["updates"] - origin_updates,
            "transitions_per_second": (committed["transitions"] - origin) / seconds,
            "rounds": len(metrics),
            "snapshots": snapshots,
            "active_transitions": active[0].transitions,
            "result_checkpoint": str(directory / "checkpoint.pt"),
        },
    )
    assert parent_hashes == {name: checksum(PARENT / name) for name in parent_hashes}
    if failure:
        raise RuntimeError(f"Stopped; see {directory / 'failure.txt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.out.resolve(), smoke=args.smoke)


if __name__ == "__main__":
    main()
