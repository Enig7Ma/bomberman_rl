"""Real worlds, full chunk saves, independent spawned runs and inference."""

import copy
import importlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from training import cli
from training.config import Curriculum, CurriculumError, parse_curriculum
from training.driver import environment, read_chunk_records, snapshots, train_run
from training.evaluate import evaluate_run
from training.frozen import frozen_name, materialise_agent, remove_frozen
from training.spec import SPECS, AgentSpec

torch = pytest.importorskip("torch")

from agent_code.dqn_agent.encoder import ENCODERS  # noqa: E402
from agent_code.dqn_agent.metrics import read_records  # noqa: E402
from agent_code.dqn_agent.network import QNetwork  # noqa: E402
from agent_code.dqn_agent.persistence import load_checkpoint, load_replay  # noqa: E402
from training import dqn  # noqa: E402


def raw_course() -> dict[str, Any]:
    return {
        "name": "d5-smoke",
        "agent": "dqn_agent",
        "params": {"warmup": 8, "replay_size": 5000, "replay_save_every": 999},
        "chunk_rounds": 1,
        "eval_every_transitions": 500,
        "evaluations": [{"preset": "coin-heaven-solo", "seeds": 1}],
        "stages": [
            {
                "name": "coins",
                "scenario": "coin-heaven",
                "lineups": [{"opponents": []}],
                "transitions": 100000,
                "rounds": 3,
                "epsilon_start": 0.3,
                "epsilon_end": 0.05,
            }
        ],
    }


def course() -> Curriculum:
    return parse_curriculum(raw_course())


def assert_equal(a: Any, b: Any) -> None:  # noqa: ANN401
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(cast(Any, a), b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in cast(dict[str, Any], a):
            assert_equal(a[key], b[key])
    elif isinstance(a, list | tuple):
        assert len(cast(list[Any], a)) == len(b)
        for left, right in zip(cast(list[Any], a), b, strict=True):
            assert_equal(left, right)
    else:
        assert a == b


@pytest.fixture(scope="module")
def smoke(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("dqn-driver")
    train_run(course(), path, 11)
    return path


def test_config_roundtrip_and_default_snapshot_interval() -> None:
    parsed = course()
    assert parse_curriculum(asdict(parsed)) == parsed
    raw = raw_course()
    del raw["eval_every_transitions"]
    assert parse_curriculum(raw).eval_every_transitions == 100000


@pytest.mark.parametrize(
    "bad",
    [
        {"transitions": 0},
        {"transitions": None},
        {"params": {"warmup": 1}},
        {"params": {"encoder": "bad"}},
        {"params": {"lr": -1}},
        {"decay_share": 0},
        {"epsilon_end": 0.9},
    ],
)
def test_invalid_dqn_stages(bad: dict[str, Any]) -> None:
    raw = raw_course()
    raw["stages"][0].update(bad)
    with pytest.raises(CurriculumError):
        parse_curriculum(raw)


def test_three_rounds_save_all_artifacts(smoke: Path) -> None:
    state = load_checkpoint(smoke / "checkpoint.pt")
    buffer, degraded = load_replay(smoke / "replay.npz", state, state["schema_id"])
    assert not degraded
    assert state["rounds_trained"] == 3
    assert len(buffer) == state["transitions"]
    assert state["learner"]["updates"] > 0
    assert len(read_records(smoke / "metrics.jsonl")) == 3
    assert len(read_chunk_records(smoke)) == 3
    assert snapshots(smoke)
    net = QNetwork.load(smoke / "q_net.npz", ENCODERS["onehot_e3"])
    for name, weights in net.arrays.items():
        np.testing.assert_array_equal(weights, state["learner"]["online"][name].numpy())
    assert list((smoke / "logs" / "chunk_000000" / "agents").rglob("*.log"))
    assert train_run(course(), smoke, 11) == []


def test_stages_change_lineup_and_coefficients_without_reset(tmp_path: Path) -> None:
    raw = raw_course()
    first = raw["stages"][0]
    first.update(rounds=1, params={"crate_aid": 1.0})
    second = copy.deepcopy(first)
    second.update(
        name="opponent",
        lineups=[{"opponents": ["peaceful_agent"]}],
        params={"lr": 0.0001, "crate_aid": 0.0},
    )
    raw["stages"].append(second)
    records = train_run(parse_curriculum(raw), tmp_path, 2)
    state = load_checkpoint(tmp_path / "checkpoint.pt")
    buffer, degraded = load_replay(tmp_path / "replay.npz", state, state["schema_id"])
    assert not degraded
    assert records[1].opponents == ("peaceful_agent",)
    assert state["transitions"] == sum(r.transitions for r in records) == len(buffer)
    assert records[1].total_updates > records[0].total_updates > 0
    assert state["stage_start"] == records[0].total_transitions
    assert state["stage_position"] == records[1].transitions
    assert state["config"]["crate_aid"] == 0
    assert state["learner"]["adam"]["param_groups"][0]["lr"] == 0.0001
    assert set(buffer.snapshot()["rows"]["stage"]) == {0, 1}
    assert train_run(parse_curriculum(raw), tmp_path, 2) == []


def test_transition_budget_finishes_current_round(tmp_path: Path) -> None:
    raw = raw_course()
    raw["chunk_rounds"] = 3
    raw["stages"][0].update(transitions=1, rounds=0)
    records = train_run(parse_curriculum(raw), tmp_path, 0)
    assert len(records) == 1 and records[0].rounds == 1
    assert records[0].transitions > 1
    state = load_checkpoint(tmp_path / "checkpoint.pt")
    buffer, _ = load_replay(tmp_path / "replay.npz", state, state["schema_id"])
    assert np.count_nonzero(buffer.snapshot()["rows"]["done"]) == 1


def test_interrupted_chunk_resumes_same_as_uninterrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    smoke: Path,
) -> None:
    original = dqn.play_round
    calls = 0

    def interrupted(world: Any) -> None:  # noqa: ANN401
        nonlocal calls
        original(world)
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted before full save")

    with monkeypatch.context() as patch:
        patch.setattr(dqn, "play_round", interrupted)
        with pytest.raises(RuntimeError, match="interrupted"):
            train_run(course(), tmp_path, 11)
    assert len(read_chunk_records(tmp_path)) == 1
    assert len(read_records(tmp_path / "metrics.jsonl")) == 2
    assert len(train_run(course(), tmp_path, 11)) == 2
    assert len(read_records(tmp_path / "metrics.jsonl")) == 3
    resumed = load_checkpoint(tmp_path / "checkpoint.pt")
    continuous = load_checkpoint(smoke / "checkpoint.pt")
    for key in (
        "learner",
        "transitions",
        "rounds_trained",
        "stage_position",
        "feature_rng",
    ):
        assert_equal(resumed[key], continuous[key])
    left, _ = load_replay(tmp_path / "replay.npz", resumed, resumed["schema_id"])
    right, _ = load_replay(smoke / "replay.npz", continuous, continuous["schema_id"])
    assert_equal(left.snapshot(), right.snapshot())


def test_save_failure_does_not_mark_chunk_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = SPECS["dqn_agent"].__class__.save_chunk

    def fail(self: AgentSpec, agent: Any) -> None:  # noqa: ANN401
        if agent.trainer.rounds_trained:
            raise OSError("save failed")
        original(self, agent)

    with monkeypatch.context() as patch:
        patch.setattr(type(SPECS["dqn_agent"]), "save_chunk", fail)
        with pytest.raises(OSError, match="save failed"):
            train_run(course(), tmp_path, 1)
    assert not (tmp_path / "chunks.jsonl").exists()
    assert load_checkpoint(tmp_path / "checkpoint.pt")["rounds_trained"] == 0
    assert len(train_run(course(), tmp_path, 1)) == 3


def test_cli_two_spawned_runs_are_independent(tmp_path: Path) -> None:
    raw = raw_course()
    raw["stages"][0]["rounds"] = 1
    raw["stages"][0]["lineups"] = [{"opponents": ["frozen"]}]
    path = tmp_path / "course.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    out = tmp_path / "runs"
    try:
        assert (
            cli.main(
                [
                    "run",
                    "--curriculum",
                    str(path),
                    "--out",
                    str(out),
                    "--runs",
                    "2",
                    "--jobs",
                    "2",
                    "--no-progress",
                ]
            )
            == 0
        )
        a, b = [load_checkpoint(out / f"run_{i}" / "checkpoint.pt") for i in (0, 1)]
        assert a["config"]["seed"] != b["config"]["seed"]
        assert not torch.equal(
            a["learner"]["online"]["W0"], b["learner"]["online"]["W0"]
        )
        for i in (0, 1):
            run = out / f"run_{i}"
            assert (run / "replay.npz").exists()
            assert len(read_records(run / "metrics.jsonl")) == 1
            name = frozen_name(i, spec=SPECS["dqn_agent"], namespace=run)
            assert read_chunk_records(run)[0].opponents == (name,)
    finally:
        for i in (0, 1):
            remove_frozen(
                frozen_name(i, spec=SPECS["dqn_agent"], namespace=out / f"run_{i}")
            )


def test_evaluation_and_frozen_load_snapshot_not_live_model(
    smoke: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import training.evaluate as evaluation
    from agent_code.dqn_agent import callbacks

    files = [smoke / name for name in ("checkpoint.pt", "replay.npz", "q_net.npz")]
    before = [p.read_bytes() for p in files]
    loaded: list[Path] = []
    original = callbacks.setup

    def setup(agent: Any) -> None:  # noqa: ANN401
        original(agent)
        assert not agent.train and agent.trainer is None
        assert agent.q_function is not None
        path = Path(os.environ["DQN_AGENT_MODEL"])
        assert path.parent.name == "snapshots"
        expected = QNetwork.load(path, agent.encoder)
        for name in expected.arrays:
            np.testing.assert_array_equal(
                agent.q_function.arrays[name], expected.arrays[name]
            )
        loaded.append(path)

    # Avoid a previously cached evaluation so this test must really execute setup.
    monkeypatch.setattr(evaluation, "EVAL_DIR", "eval-test")
    monkeypatch.setattr(callbacks, "setup", setup)
    points = evaluate_run(smoke)["coin-heaven-solo"]
    assert len(loaded) == len(snapshots(smoke)) == len(points)
    assert [p.read_bytes() for p in files] == before
    snapshot = snapshots(smoke)[0][1]
    name = frozen_name(19, spec=SPECS["dqn_agent"], namespace=tmp_path)
    try:
        model = materialise_agent(
            name, snapshot, spec=SPECS["dqn_agent"], params={}, seed=19
        )
        module = importlib.import_module(f"agent_code.{name}.callbacks")
        import logging
        from types import SimpleNamespace

        agent = SimpleNamespace(train=False, logger=logging.getLogger("frozen-test"))
        with environment(
            {f"{name.upper()}_MODEL": str(model), f"{name.upper()}_PARAMS": "{}"}
        ):
            module.setup(agent)
        assert agent.q_function is not None
        expected = QNetwork.load(snapshot, ENCODERS["onehot_e3"])
        for key in expected.arrays:
            np.testing.assert_array_equal(
                agent.q_function.arrays[key], expected.arrays[key]
            )
        assert model.read_bytes() == snapshot.read_bytes()
    finally:
        remove_frozen(name)


@pytest.mark.parametrize("kind", ["stale", "newer", "identity"])
def test_driver_rejects_mismatched_generations(
    smoke: Path,
    tmp_path: Path,
    kind: str,
) -> None:
    from agent_code.dqn_agent.persistence import save_checkpoint, save_replay

    for filename in ("run.json", "checkpoint.pt", "replay.npz", "q_net.npz"):
        shutil.copyfile(smoke / filename, tmp_path / filename)
    state = load_checkpoint(tmp_path / "checkpoint.pt")
    replay, _ = load_replay(tmp_path / "replay.npz", state, state["schema_id"])
    if kind == "stale":
        # A valid old generation, not a corrupt array disguised as stale data.
        state["transitions"] += 4
        state["stage_position"] += 4
        state["learner"]["updates"] += 1
        save_checkpoint(tmp_path / "checkpoint.pt", state)
    elif kind == "newer":
        state["transitions"] -= 1
        save_checkpoint(tmp_path / "checkpoint.pt", state)
    else:
        save_replay(
            tmp_path / "replay.npz",
            replay,
            run_id="different-run",
            transitions=state["transitions"],
            schema_id=state["schema_id"],
        )
    with pytest.raises((ValueError, RuntimeError), match="stale|newer|identity"):
        train_run(course(), tmp_path, 11)
    assert not (tmp_path / "chunks.jsonl").exists()


def test_numpy_evaluation_subprocess_never_imports_torch(
    smoke: Path, tmp_path: Path
) -> None:
    for filename in ("run.json", "metrics.jsonl"):
        shutil.copyfile(smoke / filename, tmp_path / filename)
    shutil.copytree(smoke / "snapshots", tmp_path / "snapshots")
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; "
            "from training.evaluate import evaluate_run; "
            "evaluate_run(Path(sys.argv[1])); assert 'torch' not in sys.modules",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_numpy_warm_start_is_new_training(smoke: Path, tmp_path: Path) -> None:
    raw = raw_course()
    raw["stages"][0]["rounds"] = 1
    raw["params"]["warmup"] = 100000
    source = smoke / "q_net.npz"
    before = source.read_bytes()
    train_run(parse_curriculum(raw), tmp_path, 8, init_from=source)
    state = load_checkpoint(tmp_path / "checkpoint.pt")
    assert state["rounds_trained"] == 1 and state["learner"]["updates"] == 0
    assert state["learner"]["adam"]["state"] == {}
    assert (
        read_records(tmp_path / "metrics.jsonl")[0]["resume_mode"] == "numpy-warm-start"
    )
    expected = QNetwork.load(source, ENCODERS["onehot_e3"])
    for name, weights in expected.arrays.items():
        np.testing.assert_array_equal(weights, state["learner"]["online"][name].numpy())
    assert source.read_bytes() == before


def test_checkpoint_before_chunk_marker_is_recovered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = raw_course()
    raw["stages"][0]["rounds"] = 1
    curriculum = parse_curriculum(raw)

    def fail_publish(*args: object) -> None:
        raise OSError("marker interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(dqn, "_publish", fail_publish)
        with pytest.raises(OSError, match="marker interrupted"):
            train_run(curriculum, tmp_path, 7)
    before = load_checkpoint(tmp_path / "checkpoint.pt")
    assert before["rounds_trained"] == 1
    assert not (tmp_path / "chunks.jsonl").exists()
    assert train_run(curriculum, tmp_path, 7) == []
    after = load_checkpoint(tmp_path / "checkpoint.pt")
    assert_equal(before, after)
    assert len(read_chunk_records(tmp_path)) == 1
    assert len(read_records(tmp_path / "metrics.jsonl")) == 1
    assert len(snapshots(tmp_path)) == 1
