"""A new curriculum branches state, not only inference weights."""

import copy
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("torch")

from agent_code.dqn_agent.persistence import load_checkpoint, load_replay  # noqa: E402
from tests.test_training_dqn import assert_equal, course  # noqa: E402
from training.continuation import fork_curriculum  # noqa: E402
from training.driver import train_run  # noqa: E402


@pytest.mark.parametrize("lr", [3e-4, 1e-4])
def test_full_fork_and_resume_preserve_state(tmp_path: Path, lr: float) -> None:
    original = course()
    original = replace(original, stages=(replace(original.stages[0], rounds=1),))
    parent, child = tmp_path / "parent", tmp_path / "child"
    train_run(original, parent, 11)
    before = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in parent.glob("*.npz")
    }
    stored = load_checkpoint(parent / "checkpoint.pt")
    following = replace(
        original,
        name="next",
        params={**original.params, "lr": lr},
        stages=(replace(original.stages[0], rounds=2),),
    )
    trainer = fork_curriculum(parent, child, following, 11)
    forked = load_checkpoint(child / "checkpoint.pt")
    expected_learner = copy.deepcopy(stored["learner"])
    for group in expected_learner["adam"]["param_groups"]:
        group["lr"] = lr
    assert_equal(expected_learner, forked["learner"])
    assert all(group["lr"] == lr for group in trainer.learner.optimizer.param_groups)
    for key in ("feature_rng", "transitions", "rounds_trained"):
        assert_equal(stored[key], forked[key])
    assert trainer.stage_position == 0
    assert trainer.stage_start == stored["transitions"]
    assert trainer.config.stage == stored["config"]["stage"] + 1
    assert trainer.learner.epsilon(0, trainer.config.stage_transitions) == 0.3
    replay, degraded = load_replay(child / "replay.npz", forked, forked["schema_id"])
    assert not degraded and len(replay) == len(trainer.replay)
    parent_replay, _ = load_replay(parent / "replay.npz", stored, stored["schema_id"])
    assert_equal(parent_replay.snapshot(), replay.snapshot())
    train_run(following, child, 11)
    complete = load_checkpoint(child / "checkpoint.pt")
    assert all(
        group["lr"] == lr for group in complete["learner"]["adam"]["param_groups"]
    )
    assert complete["transitions"] > forked["transitions"]
    assert complete["stage_start"] == forked["transitions"]
    assert complete["rounds_trained"] == stored["rounds_trained"] + 2
    assert train_run(following, child, 11) == []
    assert_equal(
        complete["learner"], load_checkpoint(child / "checkpoint.pt")["learner"]
    )
    assert before == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in parent.glob("*.npz")
    }
    assert_equal(stored, load_checkpoint(parent / "checkpoint.pt"))
    with pytest.raises(FileExistsError):
        fork_curriculum(parent, child, following, 11)

