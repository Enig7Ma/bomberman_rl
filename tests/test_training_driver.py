"""Tests for the curriculum driver, frozen opponents, evaluation and the CLI."""

import importlib
import json
import os
from collections.abc import Iterator, Sequence
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from agent_code.tabular_q_agent.features import ENCODINGS
from agent_code.tabular_q_agent.metrics import read_records
from agent_code.tabular_q_agent.qtable import QTable
from training import cli
from training.config import Curriculum, Evaluation, Lineup, Stage
from training.driver import (
    CHUNKS_FILE,
    METRICS_FILE,
    MODEL_FILE,
    RUN_FILE,
    ChunkRecord,
    environment,
    plan_chunks,
    read_chunk_records,
    snapshots,
    train_run,
    world_seed,
)
from training.evaluate import evaluate_run
from training.frozen import AGENT_CODE, frozen_name, materialise_frozen, remove_frozen

E1 = ENCODINGS["E1"]


def stage(
    name: str = "coins",
    scenario: str = "coin-heaven",
    *,
    rounds: int = 3,
    lineups: Sequence[Sequence[str]] = ((),),
    replay_share: float = 0.0,
    epsilon: tuple[float, float] = (0.3, 0.05),
) -> Stage:
    return Stage(
        name=name,
        scenario=scenario,
        lineups=tuple(Lineup(tuple(opponents)) for opponents in lineups),
        rounds=rounds,
        epsilon_start=epsilon[0],
        epsilon_end=epsilon[1],
        decay_share=0.6,
        replay_share=replay_share,
    )


def curriculum(
    *stages: Stage,
    chunk_rounds: int = 1,
    eval_every: int = 1,
    evaluations: Sequence[Evaluation] = (),
) -> Curriculum:
    return Curriculum(
        name="test",
        params={"encoding": "E1"},
        chunk_rounds=chunk_rounds,
        eval_every=eval_every,
        evaluations=tuple(evaluations),
        stages=stages,
    )


# --- the chunk plan ----------------------------------------------------------


def test_world_seeds_are_disjoint_per_run() -> None:
    assert world_seed(0, 0) == 100_000
    assert world_seed(3, 7) == 130_007
    with pytest.raises(ValueError):
        world_seed(0, 10_000)


def test_the_plan_covers_every_round_in_chunks() -> None:
    plan = plan_chunks(
        curriculum(
            stage(rounds=4), stage("crates", "loot-crate", rounds=6), chunk_rounds=2
        ),
        run_seed=0,
    )
    assert [chunk.stage for chunk in plan] == ["coins"] * 2 + ["crates"] * 3
    assert [chunk.index for chunk in plan] == list(range(5))
    assert [chunk.seed for chunk in plan] == [100_000 + i for i in range(5)]
    assert [chunk.scenario for chunk in plan] == ["coin-heaven"] * 2 + [
        "loot-crate"
    ] * 3


def test_epsilon_follows_the_stage_schedule() -> None:
    # 10 rounds, decay over the first 6; chunks start at rounds 0, 2, 4, 6, 8.
    plan = plan_chunks(curriculum(stage(rounds=10), chunk_rounds=2), run_seed=0)
    expected = [0.3, 0.3 - 0.25 / 3, 0.3 - 0.5 / 3, 0.05, 0.05]
    assert [chunk.epsilon for chunk in plan] == pytest.approx(expected)


def test_replays_come_from_earlier_stages_only() -> None:
    plan = plan_chunks(
        curriculum(
            stage("coins", "coin-heaven", rounds=4),
            stage("crates", "loot-crate", rounds=4),
            stage(
                "fight",
                "classic",
                rounds=200,
                lineups=(("peaceful_agent",),),
                replay_share=0.5,
            ),
            chunk_rounds=2,
        ),
        run_seed=1,
    )
    assert not any(chunk.replay for chunk in plan[:4])
    fight = [chunk for chunk in plan if chunk.stage == "fight"]
    replays = [chunk for chunk in fight if chunk.replay]
    assert 0 < len(replays) < len(fight)
    assert {chunk.source for chunk in replays} == {"coins", "crates"}
    scenarios = {"coins": "coin-heaven", "crates": "loot-crate"}
    assert all(c.scenario == scenarios[c.source] and c.opponents == () for c in replays)
    assert all(c.label == f"fight/replay:{c.source}" for c in replays)
    originals = [chunk for chunk in fight if not chunk.replay]
    assert all(
        c.scenario == "classic" and c.opponents == ("peaceful_agent",)
        for c in originals
    )


def test_the_plan_depends_on_the_run_seed_only() -> None:
    lineups = (Lineup(("peaceful_agent",), 3.0), Lineup(("random_agent",), 1.0))
    weighted = replace(stage(rounds=400), lineups=lineups)
    course = curriculum(weighted)
    plan = plan_chunks(course, run_seed=4)
    assert plan_chunks(course, run_seed=4) == plan
    peaceful = sum(chunk.opponents == ("peaceful_agent",) for chunk in plan)
    assert 0.65 < peaceful / len(plan) < 0.85  # weight 3 : 1
    other = plan_chunks(course, run_seed=5)
    assert [c.opponents for c in other] != [c.opponents for c in plan]


def test_environment_sets_and_restores_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TQ_TEST_KEPT", "old")
    monkeypatch.delenv("TQ_TEST_NEW", raising=False)
    with environment({"TQ_TEST_KEPT": "new", "TQ_TEST_NEW": "x"}):
        assert (os.environ["TQ_TEST_KEPT"], os.environ["TQ_TEST_NEW"]) == ("new", "x")
    assert os.environ["TQ_TEST_KEPT"] == "old"
    assert "TQ_TEST_NEW" not in os.environ


# --- a real run ----------------------------------------------------------------

SMOKE = curriculum(
    stage(rounds=3),
    evaluations=(Evaluation("coin-heaven-solo", seeds=1, seed_start=500),),
)


@pytest.fixture(scope="module")
def smoke_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, list[ChunkRecord]]:
    run_dir = tmp_path_factory.mktemp("smoke") / "run_0"
    return run_dir, train_run(SMOKE, run_dir, 0)


def test_a_run_trains_checkpoints_and_snapshots(
    smoke_run: tuple[Path, list[ChunkRecord]],
) -> None:
    run_dir, records = smoke_run
    assert [record.rounds_trained for record in records] == [1, 2, 3]
    assert QTable.load(run_dir / MODEL_FILE, E1).meta["rounds_trained"] == 3
    metrics = read_records(run_dir / METRICS_FILE)
    assert [m.rounds_trained for m in metrics] == [1, 2, 3]
    assert {m.stage for m in metrics} == {"coins"}
    assert read_chunk_records(run_dir) == records
    assert [record.seed for record in records] == [100_000, 100_001, 100_002]
    assert all(r.rounds_per_second > 0 and r.steps_per_second > 0 for r in records)
    assert [rounds for rounds, _ in snapshots(run_dir)] == [1, 2, 3]
    stored = json.loads((run_dir / RUN_FILE).read_text())
    assert stored["run_seed"] == 0
    assert stored["curriculum"]["name"] == "test"


def test_a_finished_run_resumes_as_a_no_op(
    smoke_run: tuple[Path, list[ChunkRecord]],
) -> None:
    run_dir, _ = smoke_run
    assert train_run(SMOKE, run_dir, 0) == []
    assert len(read_records(run_dir / METRICS_FILE)) == 3


def test_a_run_directory_refuses_another_curriculum_or_seed(
    smoke_run: tuple[Path, list[ChunkRecord]],
) -> None:
    run_dir, _ = smoke_run
    with pytest.raises(RuntimeError):
        train_run(replace(SMOKE, name="other"), run_dir, 0)
    with pytest.raises(RuntimeError):
        train_run(SMOKE, run_dir, 1)


def test_snapshots_are_evaluated_into_a_learning_curve(
    smoke_run: tuple[Path, list[ChunkRecord]],
) -> None:
    run_dir, _ = smoke_run
    points = evaluate_run(run_dir)["coin-heaven-solo"]
    assert [point.rounds_trained for point in points] == [1, 2, 3]
    assert all(point.summary.rounds == 1 for point in points)
    assert all(point.summary.code_name == "tabular_q_agent" for point in points)
    assert (run_dir / "eval" / "coin-heaven-solo_round_000003.jsonl").exists()
    curve = (run_dir / "eval" / "coin-heaven-solo_curve.md").read_text()
    rows = [line for line in curve.splitlines() if line.startswith("| ")]
    assert len(rows) == 1 + 3  # header and one row per snapshot
    # A second evaluation reuses the stored results.
    again = evaluate_run(run_dir)["coin-heaven-solo"]
    assert [p.summary for p in again] == [p.summary for p in points]


# --- frozen opponents ------------------------------------------------------------

FROZEN_SEED = 991


@pytest.fixture
def frozen() -> Iterator[str]:
    name = frozen_name(FROZEN_SEED)
    remove_frozen(name)
    yield name
    remove_frozen(name)


def test_materialise_copies_the_code_but_not_logs_or_models(
    tmp_path: Path, frozen: str
) -> None:
    table = tmp_path / "table.npz"
    QTable.zeros(E1).save(table)
    target = materialise_frozen(frozen, table, encoding="E1")
    assert (target / "callbacks.py").is_file()
    assert (target / "core" / "safety.py").is_file()
    assert not (target / "logs").exists()
    assert sorted(p.name for p in (target / "model").iterdir()) == ["q_table.npz"]
    assert (target / "model" / "q_table.npz").read_bytes() == table.read_bytes()


def test_materialise_without_a_table_installs_an_empty_one(frozen: str) -> None:
    target = materialise_frozen(frozen, None, encoding="E1")
    assert QTable.load(target / "model" / "q_table.npz", E1).visited_states == 0


def test_only_frozen_directories_are_touched() -> None:
    with pytest.raises(ValueError):
        materialise_frozen("rule_based_agent", None, encoding="E1")
    with pytest.raises(ValueError):
        remove_frozen("bfs_agent")
    assert (AGENT_CODE / "bfs_agent" / "callbacks.py").is_file()


def test_a_frozen_opponent_plays_the_newest_snapshot(
    tmp_path: Path, frozen: str
) -> None:
    run_dir = tmp_path / "run"
    train_run(curriculum(stage(rounds=2, lineups=(("frozen",),))), run_dir, FROZEN_SEED)

    copy = AGENT_CODE / frozen
    newest = snapshots(run_dir)[-1][1]
    assert (copy / "model" / "q_table.npz").read_bytes() == newest.read_bytes()
    assert all(m.opponents == [frozen] for m in read_records(run_dir / METRICS_FILE))
    config = importlib.import_module(f"agent_code.{frozen}.config")
    assert config.ENV_VAR == f"{frozen.upper()}_PARAMS"


# --- command line -----------------------------------------------------------------


def test_the_cli_runs_independent_seeds_in_parallel(tmp_path: Path) -> None:
    path = tmp_path / "curriculum.json"
    path.write_text(json.dumps(asdict(curriculum(stage(rounds=1)))))
    out = tmp_path / "runs"
    argv = ["run", "--curriculum", str(path), "--out", str(out)]
    argv += ["--runs", "2", "--jobs", "2", "--seed", "5", "--no-progress"]
    assert cli.main(argv) == 0

    first = [read_chunk_records(out / f"run_{seed}")[0].seed for seed in (5, 6)]
    assert first == [world_seed(5, 0), world_seed(6, 0)]
    assert (out / "run_6" / CHUNKS_FILE).exists()
    assert cli.main(["evaluate", str(out), "--no-progress"]) == 0
    assert cli.main(["evaluate", str(tmp_path / "nothing")]) == 2
