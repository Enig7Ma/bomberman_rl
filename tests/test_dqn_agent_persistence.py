"""D4 weight-only loading, replay consistency, atomic writes and probe checks."""

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from agent_code.dqn_agent.encoder import OneHotE3
from agent_code.dqn_agent.network import QNetwork
from agent_code.dqn_agent.probe import Probe, check_q
from tests.test_dqn_agent_train import configure, script, trainer
from tournament.engine import quiet_logging, reset_framework_logging
from training.world import create_training_world, play_round

torch = pytest.importorskip("torch")


def test_stale_replay_is_explicit_and_stays_degraded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configure(monkeypatch, tmp_path, replay_save_every=50)
    with quiet_logging():
        world = create_training_world(("dqn_agent", "peaceful_agent"), "empty", 0)
        subject = trainer(world)
        script(monkeypatch, subject)
        for _ in range(2):
            play_round(world)
        reset_framework_logging()
        restored = trainer(create_training_world(("dqn_agent",), "empty", 0))
        assert restored.transitions == 800 and len(restored.replay) == 400
        assert (
            restored.resume_mode == "stale-replay-resume" and not restored.exact_history
        )
        restored.save(full=True)
        reset_framework_logging()
        again = trainer(create_training_world(("dqn_agent",), "empty", 0))
        assert not again.exact_history
        reset_framework_logging()


@pytest.mark.parametrize(
    "fault", ["missing", "newer", "identity", "schema", "position", "config", "corrupt"]
)
def test_inconsistent_files_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fault: str
) -> None:
    configure(monkeypatch, tmp_path)
    with quiet_logging():
        world = create_training_world(("dqn_agent",), "empty", 0)
        subject = trainer(world)
        script(monkeypatch, subject, ["BOMB"])
        play_round(world)
        reset_framework_logging()
        path = tmp_path / "replay.npz"
        if fault == "missing":
            path.unlink()
        elif fault == "corrupt":
            (tmp_path / "checkpoint.pt").write_bytes(b"broken checkpoint")
        elif fault == "config":
            configure(monkeypatch, tmp_path, lr=0.001)
        else:
            with np.load(path, allow_pickle=False) as archive:
                rows = archive["rows"].copy()
                meta = json.loads(str(archive["meta"].item()))
            key, value = {
                "newer": ("transitions", 6),
                "identity": ("run_id", "other"),
                "schema": ("schema_id", "other"),
                "position": ("write", 0),
            }[fault]
            meta[key] = value
            np.savez(path, rows=rows, meta=np.array(json.dumps(meta)))
        with pytest.raises((ValueError, FileNotFoundError, pickle.UnpicklingError)):
            create_training_world(("dqn_agent",), "empty", 0)
        reset_framework_logging()


def test_numpy_warm_start_has_fresh_optimizer_and_counters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configure(monkeypatch, tmp_path)
    net = QNetwork.random(OneHotE3(), 99)
    net.meta.update({"updates": 100, "transitions": 5000, "rounds_trained": 4})
    net.save(tmp_path / "q_net.npz")
    with quiet_logging():
        subject = trainer(create_training_world(("dqn_agent",), "empty", 0))
        assert subject.resume_mode == "numpy-warm-start"
        assert (
            subject.transitions
            == subject.stage_position
            == subject.rounds_trained
            == subject.learner.updates
            == 0
        )
        assert len(subject.replay) == 0 and not subject.learner.optimizer.state
        for name, array in net.arrays.items():
            np.testing.assert_array_equal(subject.learner.export().arrays[name], array)
        assert all(
            torch.equal(a, b)
            for a, b in zip(
                subject.learner.online.parameters(),
                subject.learner.target.parameters(),
                strict=True,
            )
        )
        reset_framework_logging()


def test_checkpoint_load_uses_weights_only_and_paths_ignore_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from agent_code.dqn_agent import persistence

    configure(monkeypatch, tmp_path)
    with quiet_logging():
        subject = trainer(create_training_world(("dqn_agent",), "empty", 0))
        subject.save(full=True)
        reset_framework_logging()
    real_load = torch.load
    calls: list[dict[str, Any]] = []

    def load(*args: object, **kwargs: object) -> object:
        calls.append(dict(kwargs))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(persistence.torch, "load", load)
    monkeypatch.chdir(tmp_path)
    state = persistence.load_checkpoint(
        subject.agent.model_file.parent / "checkpoint.pt"
    )
    assert state["transitions"] == 0
    assert calls[0]["weights_only"] is True


def test_atomic_failed_save_preserves_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from agent_code.dqn_agent import persistence

    path = tmp_path / "checkpoint.pt"
    state = {"format_version": 1, "value": 5}
    persistence.save_checkpoint(path, state)
    before = path.read_bytes()

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(persistence.os, "replace", fail)
    with pytest.raises(OSError):
        persistence.save_checkpoint(path, {**state, "value": 8})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_probe_histogram_churn_guard_and_load(tmp_path: Path) -> None:
    net = QNetwork.random(OneHotE3(), 0)
    for array in net.arrays.values():
        array.fill(0)
    net.arrays["b2"][:] = [1, 2, 40, 0, 0, 0]
    x = np.zeros((3, 32), dtype=np.float32)
    masks = np.zeros((3, 6), dtype=bool)
    masks[:, :2] = True
    path = tmp_path / "probe.npz"
    np.savez(path, x=x, masks=masks, schema_id=np.array(OneHotE3().schema_id))
    probe = Probe.load(path, OneHotE3().schema_id, 32)
    report = probe.measure(net)
    assert report["mean_max_q"] == report["max_q"] == 2
    assert report["greedy_histogram"] == [0, 3, 0, 0, 0, 0]
    assert report["policy_churn"] is None
    net.arrays["b2"][0] = 3
    assert probe.measure(net)["policy_churn"] == 1
    assert probe.measure(net)["policy_churn"] == 0
    for value in (float("nan"), float("inf"), 50.01, -50.01):
        with pytest.raises(FloatingPointError):
            check_q(np.array([value], dtype=np.float32))
    check_q(np.array([50, -50], dtype=np.float32))
    with pytest.raises(ValueError):
        Probe.load(path, "other", 32)


def test_training_probe_interval_and_guard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "probe.npz"
    np.savez(
        path,
        x=np.zeros((2, 32), dtype=np.float32),
        masks=np.ones((2, 6), dtype=bool),
        schema_id=np.array(OneHotE3().schema_id),
    )
    configure(
        monkeypatch,
        tmp_path,
        warmup=0,
        train_every=1,
        probe_every=1,
        probe_path=str(path),
    )
    with quiet_logging():
        world = create_training_world(("dqn_agent",), "empty", 0)
        subject = trainer(world)
        script(monkeypatch, subject, ["BOMB"])
        play_round(world)
        assert len(subject.last_record["probe"]) == 5
        assert all(r["status"] == "measured" for r in subject.last_record["probe"])
        assert subject.last_record["updates"] == 5
        reset_framework_logging()
        restored = trainer(create_training_world(("dqn_agent",), "empty", 0))
        assert restored.probe is not None and subject.probe is not None
        assert restored.probe.previous == subject.probe.previous
        assert restored.probe.measure(restored.learner.online)["policy_churn"] == 0
        reset_framework_logging()


def test_midround_save_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configure(monkeypatch, tmp_path)
    with quiet_logging():
        world = create_training_world(("dqn_agent",), "empty", 0)
        subject = trainer(world)
        script(monkeypatch, subject, ["BOMB"])
        world.new_round()
        world.do_step()
        with pytest.raises(RuntimeError, match="completed round"):
            subject.save(full=True)
        assert not (tmp_path / "checkpoint.pt").exists()
        reset_framework_logging()
