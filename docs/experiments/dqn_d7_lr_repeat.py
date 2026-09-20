"""One authorized stage2 LR comparison, using the original driver and evaluation."""

import argparse
import copy
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from agent_code.dqn_agent.learner import Learner, UpdateStats
from agent_code.dqn_agent.persistence import load_checkpoint, load_replay
from agent_code.dqn_agent.replay import ReplayBatch
from agent_code.dqn_agent.train import Trainer
from docs.experiments import dqn_d7
from docs.experiments.dqn_d6 import checksum, dump
from docs.experiments.dqn_d7_diagnose import collect
from tests.test_training_dqn import assert_equal
from training.config import Curriculum, load_curriculum
from training.continuation import fork_curriculum
from training.driver import train_run
from training.spec import AgentSpec

CONFIG = Path("docs/experiments/dqn_d7_lr1e4.json")
CONTROL = Path("results/dqn/d7_stage2_seed0_20260918")


def run(directory: Path) -> None:
    control, treatment = load_curriculum(dqn_d7.CONFIG), load_curriculum(CONFIG)
    expected = asdict(control)
    expected["name"] = treatment.name
    expected["params"]["lr"] = 1e-4
    assert expected == asdict(treatment)
    original_result = json.loads((CONTROL / "summary.json").read_text())
    assert all(
        checksum(dqn_d7.PARENT / name) == sha
        for name, sha in original_result["parent_sha256"].items()
    )
    paths = [
        p
        for root in (CONTROL, dqn_d7.PARENT)
        for p in root.rglob("*")
        if p.is_file() and p.suffix in (".pt", ".npz", ".json", ".jsonl")
    ]
    hashes = {str(p): checksum(p) for p in paths}
    original_update = Learner.update
    recorded = False

    def checked_fork(
        parent: Path, destination: Path, course: Curriculum, seed: int
    ) -> Trainer:
        assert parent.resolve() == dqn_d7.PARENT.resolve() and seed == 0
        trainer = fork_curriculum(parent, destination, course, seed)
        stored = load_checkpoint(parent / "checkpoint.pt")
        # configure_stage already assigns LR after restoring Adam. Assign explicitly
        # here too, then compare every other optimizer/learner field with D6.
        for group in trainer.learner.optimizer.param_groups:
            group["lr"] = 1e-4
        expected_learner = copy.deepcopy(stored["learner"])
        for group in expected_learner["adam"]["param_groups"]:
            group["lr"] = 1e-4
        assert_equal(expected_learner, trainer.learner.training_state())
        assert_equal(stored["feature_rng"], trainer.agent.rng.getstate())
        replay, degraded = load_replay(
            parent / "replay.npz", stored, stored["schema_id"]
        )
        assert not degraded
        assert_equal(replay.snapshot(), trainer.replay.snapshot())
        assert trainer.transitions == stored["transitions"] == 50047
        assert trainer.stage_position == 0 and trainer.stage_start == 50047
        trainer.save(full=True)
        dump(
            destination / "comparison_protocol.json",
            {
                "config_difference": {
                    "params.lr": [3e-4, 1e-4],
                    "name": [control.name, treatment.name],
                },
                "inherited_state_equal_except_lr": True,
                "before_sha256": hashes,
                "parent_checkpoint_sha256": checksum(parent / "checkpoint.pt"),
                "optimizer_lr_after_load": [
                    g["lr"] for g in trainer.learner.optimizer.param_groups
                ],
            },
        )
        shutil.copyfile(Path(__file__), destination / "lr_repeat_source.py")
        return trainer

    def checked_update(learner: Learner, batch: ReplayBatch) -> UpdateStats:
        nonlocal recorded
        actual = [g["lr"] for g in learner.optimizer.param_groups]
        assert all(lr == 1e-4 for lr in actual)
        if not recorded:
            dump(
                directory / "first_update_lr.json",
                {"updates_before": learner.updates, "lr": actual},
            )
            recorded = True
        return original_update(learner, batch)

    with (
        patch.object(dqn_d7, "CONFIG", CONFIG),
        patch.object(dqn_d7, "fork_curriculum", checked_fork),
        patch.object(Learner, "update", checked_update),
    ):
        dqn_d7.run(directory)
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["failure"] is None and recorded
    # Same definitions as the control diagnostics: three paired loot maps.
    counts = tuple(
        row["global_transitions"]
        for row in summary["snapshots"][1:]
        if row["stage_transitions"] < 150000 or row == summary["snapshots"][-1]
    )
    collect(
        directory / "diagnostics",
        root=directory,
        counts=counts,
        scenarios=("loot-crate",),
    )
    assert hashes == {str(p): checksum(p) for p in paths}
    dump(
        directory / "originals_unchanged.json",
        {"verified_files": len(paths), "unchanged": True},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume:
        resume(args.out.resolve())
    else:
        run(args.out.resolve())


def resume(directory: Path) -> None:
    """Continue a session-interrupted run from its existing strict driver cursor."""
    if (directory / "failure.txt").exists() or (directory / "summary.json").exists():
        raise ValueError("refuse diagnostic failure or already completed run")
    course = load_curriculum(CONFIG)
    stored = load_checkpoint(directory / "checkpoint.pt")
    _, degraded = load_replay(directory / "replay.npz", stored, stored["schema_id"])
    assert not degraded and stored["exact_history"]
    assert stored["config"]["lr"] == 1e-4
    assert all(g["lr"] == 1e-4 for g in stored["learner"]["adam"]["param_groups"])
    protocol = json.loads((directory / "comparison_protocol.json").read_text())
    hashes = protocol["before_sha256"]
    assert all(checksum(Path(p)) == sha for p, sha in hashes.items())
    snapshots = json.loads((directory / "snapshots_summary.json").read_text())
    dump(
        directory / "resume_provenance.json",
        {
            "checkpoint_sha256": checksum(directory / "checkpoint.pt"),
            "transitions": stored["transitions"],
            "stage_position": stored["stage_position"],
            "updates": stored["learner"]["updates"],
            "lr": 1e-4,
            "reason": "session interrupted; no running process; "
            "strict full pair verified",
        },
    )
    shutil.copyfile(Path(__file__), directory / "resume_source.py")
    original_save, original_update = AgentSpec.save_chunk, Learner.update
    eval_seconds = 0.0

    def checked_update(learner: Learner, batch: ReplayBatch) -> UpdateStats:
        assert all(g["lr"] == 1e-4 for g in learner.optimizer.param_groups)
        return original_update(learner, batch)

    def save_hook(spec: AgentSpec, agent: object) -> None:
        nonlocal eval_seconds
        trainer = cast(Trainer, cast(Any, agent).trainer)
        assert trainer.driver_state is not None
        if not trainer.driver_state["records"][-1]["snapshot"]:
            original_save(spec, agent)
            return
        assert trainer.probe is not None
        probe = trainer.probe.measure(trainer.learner.online)
        trainer.save(full=True)
        path = directory / "checkpoints" / f"transition_{trainer.transitions:09d}"
        path.mkdir(parents=True, exist_ok=True)
        for name in ("checkpoint.pt", "replay.npz", "q_net.npz"):
            shutil.copyfile(directory / name, path / name)
        dump(path / "probe.json", probe)
        before = {
            n: checksum(directory / n)
            for n in ("checkpoint.pt", "replay.npz", "q_net.npz")
        }
        started = time.perf_counter()
        evaluation = dqn_d7.evaluate(path / "q_net.npz", path / "evaluation")
        eval_seconds += time.perf_counter() - started
        assert before == {n: checksum(directory / n) for n in before}
        snapshots.append(
            {
                "global_transitions": trainer.transitions,
                "stage_transitions": trainer.stage_position,
                "updates": trainer.learner.updates,
                "checkpoint": str(path / "checkpoint.pt"),
                "checkpoint_sha256": checksum(path / "checkpoint.pt"),
                "probe": probe,
                "evaluation": evaluation,
            }
        )
        dump(directory / "snapshots_summary.json", snapshots)
        print(json.dumps(snapshots[-1]), flush=True)

    started = time.perf_counter()
    try:
        with (
            patch.object(AgentSpec, "save_chunk", save_hook),
            patch.object(Learner, "update", checked_update),
        ):
            train_run(course, directory, 0)
    except Exception as exc:
        (directory / "failure.txt").write_text(repr(exc), encoding="utf-8")
        raise
    elapsed = time.perf_counter() - started
    final = load_checkpoint(directory / "checkpoint.pt")
    assert final["driver_state"]["next_stage"] == 1
    dump(
        directory / "summary.json",
        {
            "failure": None,
            "resumed": True,
            "resume_seconds_including_evaluation": elapsed,
            "resume_evaluation_seconds": eval_seconds,
            "training_seconds": None,
            "timing_limitation": "original wall timer lost; "
            "suspension contaminates chunk time; report separately",
            "stage_transitions": final["stage_position"],
            "global_transitions": final["transitions"],
            "stage_updates": final["learner"]["updates"] - 11262,
            "global_updates": final["learner"]["updates"],
            "snapshots": snapshots,
            "rounds": final["rounds_trained"] - 319,
        },
    )
    counts = (snapshots[1]["global_transitions"], final["transitions"])
    collect(
        directory / "diagnostics",
        root=directory,
        counts=counts,
        scenarios=("loot-crate",),
    )
    assert all(checksum(Path(p)) == sha for p, sha in hashes.items())
    dump(
        directory / "originals_unchanged.json",
        {"verified_files": len(hashes), "unchanged": True},
    )


if __name__ == "__main__":
    main()
