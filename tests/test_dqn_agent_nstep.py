"""Ordered returns, raw reward recomputation, legacy migration and resume."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import events as e
from agent_code.dqn_agent.config import Config
from agent_code.dqn_agent.nstep import NStep, convert_replay
from agent_code.dqn_agent.replay import ReplayBuffer, ReplayTransition
from agent_code.dqn_agent.rewards import Rewards


def step(i: int, *, done: bool = False, round_id: int = 1) -> ReplayTransition:
    return ReplayTransition(
        np.full(32, i, dtype=np.float32),
        2,
        float(i + 1),
        i % 3,
        int(done),
        1 / (i + 1),
        1 / (i + 2),
        np.full(32, i + 1, dtype=np.float32),
        np.array([True, False, True, False, False, False]),
        done,
        0,
        round_id,
        i,
    )


@pytest.mark.parametrize("length", [1, 2, 3])
@pytest.mark.parametrize(
    "gamma,c_coin,crate_aid,death_aid",
    [
        (0.99, 0.5, 0, 0),
        (0.5, 2, 0.3, -4),
        (0, 1, 2, -3),
        (1, 0, 1, -1),
    ],
)
def test_terminal_tails_match_discounted_original_rewards(
    length: int,
    gamma: float,
    c_coin: float,
    crate_aid: float,
    death_aid: float,
) -> None:
    queue = NStep(3, 32)
    source = [step(i, done=i == length - 1) for i in range(length)]
    emitted: list[ReplayTransition] = []
    for t in source:
        emitted.extend(queue.push(t))
    assert not queue.pending and len(emitted) == length
    rewards = Rewards(gamma, c_coin, crate_aid, death_aid)
    for start, t in enumerate(emitted):
        replay = ReplayBuffer(1, 32)
        replay.push(t)
        batch = replay.sample(
            1,
            np.random.default_rng(0),
            gamma=gamma,
            c_coin=c_coin,
            crate_aid=crate_aid,
            death_aid=death_aid,
        )
        expected = 0.0
        for j, s in enumerate(source[start:]):
            events = [e.COIN_COLLECTED] * int(s.base) + [e.CRATE_DESTROYED] * s.crates
            if s.done:
                events += [e.GOT_KILLED, e.KILLED_SELF]
            r = (
                rewards.base(events)
                + rewards.aids(events)
                + rewards.shaping(
                    rewards.potential(s.transition_id),
                    0 if s.done else rewards.potential(s.transition_id + 1),
                )
            )
            expected += gamma**j * r
        assert batch.r[0] == pytest.approx(expected, abs=2e-6)
        assert batch.k.tolist() == [length - start]
        assert batch.done.all() and not batch.mask_next.any()
        assert not batch.x_next.any()


def test_three_step_double_dqn_target_and_canonical_successor() -> None:
    torch = pytest.importorskip("torch")
    from agent_code.dqn_agent.encoder import OneHotE3
    from agent_code.dqn_agent.learner import Learner

    queue = NStep(3, 32)
    assert queue.push(step(0)) == []
    assert queue.push(step(1)) == []
    last = replace(
        step(2), mask_next=np.array([False, False, True, False, True, False])
    )
    ready = queue.push(last)
    assert len(ready) == 1 and len(queue.pending) == 2
    assert ready[0].a == 2
    np.testing.assert_array_equal(ready[0].x, step(0).x)
    np.testing.assert_array_equal(ready[0].x_next, last.x_next)
    np.testing.assert_array_equal(ready[0].mask_next, last.mask_next)
    replay = ReplayBuffer(1, 32)
    replay.push(ready[0])
    learner = Learner(OneHotE3(), Config(gamma=0.5, n_step=3, c_coin=0, seed=0))
    with torch.no_grad():
        for net in (learner.online, learner.target):
            for p in net.parameters():
                p.zero_()
        learner.online.b2.copy_(torch.tensor([0, 100, 5, 0, 2, 0]))
        learner.target.b2.copy_(torch.tensor([0, 999, 7, 0, 20, 0]))
    batch = replay.sample(1, np.random.default_rng(0), gamma=0.5)
    assert learner.targets(batch).item() == pytest.approx(
        1 + 0.5 * 2 + 0.25 * 3 + 0.125 * 7
    )
    terminal = replace(
        batch, done=np.ones(1, dtype=bool), mask_next=np.zeros((1, 6), dtype=bool)
    )
    assert learner.targets(terminal).item() == batch.r[0]


def test_no_cross_episode_gap_or_state_discontinuity() -> None:
    queue = NStep(3, 32)
    queue.push(step(0))
    for bad in (step(1, round_id=2), step(2), replace(step(1), x=step(9).x)):
        with pytest.raises(ValueError, match="discontinuous"):
            queue.push(bad)
        assert len(queue.pending) == 1
    assert len(queue.push(step(1, done=True))) == 2
    assert queue.push(step(2, round_id=2)) == []


def test_one_step_is_identical_and_inputs_are_owned() -> None:
    queue = NStep(1, 32)
    direct, accumulated = ReplayBuffer(4, 32), ReplayBuffer(4, 32)
    for i in range(4):
        t = step(i, done=i == 3)
        direct.push(t)
        ready = queue.push(t)
        assert len(ready) == 1 and not queue.pending
        accumulated.push(ready[0])
        t.x.fill(100)
    np.testing.assert_array_equal(
        direct.snapshot()["rows"], accumulated.snapshot()["rows"]
    )
    delayed = NStep(3, 32)
    t = step(0)
    delayed.push(t)
    t.x.fill(100)
    assert delayed.pending[0].x[0] == 0


def test_pending_weights_only_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from agent_code.dqn_agent.persistence import load_checkpoint, save_checkpoint

    queue = NStep(3, 32)
    queue.push(step(0))
    queue.push(step(1))
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, {"format_version": 2, "nstep": queue.state()})
    restored = NStep.restore(load_checkpoint(path)["nstep"])
    assert restored.state() == queue.state()
    a, b = ReplayBuffer(4, 32), ReplayBuffer(4, 32)
    for output, q in ((a, queue), (b, restored)):
        for t in q.push(step(2, done=True)):
            output.push(t)
    np.testing.assert_array_equal(a.snapshot()["rows"], b.snapshot()["rows"])


def test_trainer_checkpoint_preserves_pending_and_real_counter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from tests.test_dqn_agent_train import configure, trainer
    from tournament.engine import quiet_logging, reset_framework_logging
    from training.world import create_training_world

    configure(monkeypatch, tmp_path, n_step=3)
    with quiet_logging():
        subject = trainer(create_training_world(("dqn_agent",), "empty", 0))
        for i in range(2):
            subject.nstep.push(step(i))
        subject.transitions = subject.stage_position = 2
        subject.save(full=True)
        reset_framework_logging()
        restored = trainer(create_training_world(("dqn_agent",), "empty", 0))
        assert restored.transitions == 2 and len(restored.replay) == 0
        assert restored.nstep.state() == subject.nstep.state()
        for ready in restored.nstep.push(step(2, done=True)):
            restored.replay.push(ready)
        assert len(restored.replay) == 3 and not restored.nstep.pending
        reset_framework_logging()


def test_conversion_preserves_wrapped_order_and_rejects_missing_data() -> None:
    replay = ReplayBuffer(4, 32)
    for i in range(7):
        replay.push(step(i, done=i in (2, 6), round_id=1 if i <= 2 else 2))
    converted = convert_replay(replay)
    a, b = replay.snapshot(), converted.snapshot()
    assert a["write"] == b["write"] == 3
    np.testing.assert_array_equal(
        a["rows"]["transition_id"], b["rows"]["transition_id"]
    )
    assert dict(zip(b["rows"]["transition_id"], b["rows"]["k"], strict=True)) == {
        3: 3,
        4: 3,
        5: 2,
        6: 1,
    }
    with pytest.raises(ValueError, match="one-step"):
        convert_replay(converted)
    incomplete = ReplayBuffer(4, 32)
    incomplete.push(step(0))
    with pytest.raises(ValueError, match="without terminal"):
        convert_replay(incomplete)
    incomplete.push(step(2, done=True))
    with pytest.raises(ValueError, match="gap"):
        convert_replay(incomplete)


def test_legacy_replay_explicit_upgrade_and_horizon_guard(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from agent_code.dqn_agent.persistence import load_replay, save_replay

    replay = ReplayBuffer(4, 32)
    replay.push(step(0, done=True))
    state = replay.snapshot()
    names = list(state["rows"].dtype.names[:-2])
    legacy = np.empty(1, dtype=np.dtype(state["rows"].dtype.descr[:-2]))
    for name in names:
        legacy[name] = state["rows"][name]
    meta = {k: v for k, v in state.items() if k != "rows"}
    meta.update(
        format_version=1,
        run_id="test",
        transitions=1,
        schema_id="test",
        exact_history=True,
    )
    path = tmp_path / "replay.npz"
    np.savez(path, rows=legacy, meta=np.array(json.dumps(meta)))
    checkpoint: dict[str, Any] = {"run_id": "test", "transitions": 1, "config": {}}
    restored, degraded = load_replay(path, checkpoint, "test")
    assert not degraded
    np.testing.assert_array_equal(restored.snapshot()["rows"], state["rows"])
    save_replay(path, restored, run_id="test", transitions=1, schema_id="test")
    checkpoint["config"] = {"n_step": 3}
    with pytest.raises(ValueError, match="horizon"):
        load_replay(path, checkpoint, "test")


@pytest.mark.parametrize("n_step,updates", [(1, 5), (3, 3)])
def test_engine_terminal_flush_counters_and_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    n_step: int,
    updates: int,
) -> None:
    pytest.importorskip("torch")
    from tests.test_dqn_agent_train import configure, script, trainer
    from tournament.engine import quiet_logging, reset_framework_logging
    from training.world import create_training_world, play_round

    configure(monkeypatch, tmp_path, n_step=n_step, warmup=0, train_every=1)
    with quiet_logging():
        world = create_training_world(("dqn_agent",), "empty", 0)
        subject = trainer(world)
        script(monkeypatch, subject, ["BOMB"])
        play_round(world)
        assert subject.transitions == subject.stage_position == len(subject.replay) == 5
        assert subject.learner.updates == updates
        assert subject.skipped_updates == 5 - updates
        assert not subject.nstep.pending
        assert subject.replay.snapshot()["rows"]["done"].sum() == n_step
        reset_framework_logging()
        restored = trainer(create_training_world(("dqn_agent",), "empty", 0))
        assert restored.transitions == 5 and restored.learner.updates == updates
        np.testing.assert_array_equal(
            subject.replay.snapshot()["rows"], restored.replay.snapshot()["rows"]
        )
        assert restored.nstep.state() == subject.nstep.state()
        reset_framework_logging()


def test_full_parent_fork_converts_only_returns(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from agent_code.dqn_agent.persistence import load_checkpoint, load_replay
    from tests.test_training_dqn import assert_equal, course
    from training.continuation import fork_curriculum
    from training.driver import train_run

    original = course()
    original = replace(original, stages=(replace(original.stages[0], rounds=1),))
    parent, child = tmp_path / "parent", tmp_path / "child"
    train_run(original, parent, 0)
    stored = load_checkpoint(parent / "checkpoint.pt")
    next_course = replace(
        original, name="three-step", params={**original.params, "n_step": 3}
    )
    with pytest.raises(ValueError, match="explicit"):
        fork_curriculum(parent, child, next_course, 0)
    subject = fork_curriculum(parent, child, next_course, 0, convert_n_step=True)
    converted = load_checkpoint(child / "checkpoint.pt")
    assert_equal(stored["learner"], converted["learner"])
    assert_equal(stored["feature_rng"], converted["feature_rng"])
    assert subject.transitions == stored["transitions"] and subject.stage_position == 0
    replay, degraded = load_replay(
        child / "replay.npz", converted, converted["schema_id"]
    )
    assert not degraded and len(replay) == subject.transitions
    assert (
        replay.snapshot()["rows"]["k"][~replay.snapshot()["rows"]["done"]] == 3
    ).all()
    # Resume exercises actual converted checkpoint loading and a single toy round.
    train_run(next_course, child, 0)
    resumed = load_checkpoint(child / "checkpoint.pt")
    assert resumed["rounds_trained"] == stored["rounds_trained"] + 1
    assert resumed["transitions"] > stored["transitions"]
    assert_equal(stored, load_checkpoint(parent / "checkpoint.pt"))
