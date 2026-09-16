"""Tests for averaging snapshots into one table (``training.average``)."""

from pathlib import Path

import numpy as np
import pytest

from agent_code.tabular_q_agent.features import ENCODINGS
from agent_code.tabular_q_agent.qtable import QTable
from training import cli
from training.average import average_run, average_tables
from training.config import Curriculum, Lineup, Stage
from training.driver import snapshots, train_run

E1 = ENCODINGS["E1"]


def table_with(value: float, visits: int, rounds: int) -> QTable:
    table = QTable.zeros(E1)
    table.q[:] = value
    table.n[:] = visits
    table.meta = {"rounds_trained": rounds}
    return table


def test_tables_are_averaged_value_by_value(tmp_path: Path) -> None:
    paths: list[Path] = []
    for i, value in enumerate((1.0, 2.0, 6.0)):
        path = tmp_path / f"t{i}.npz"
        table_with(value, visits=i, rounds=10 * i).save(path)
        paths.append(path)

    averaged = average_tables(paths, E1)

    assert np.allclose(averaged.q, 3.0)
    assert averaged.q.dtype == np.float32
    assert int(averaged.n[0, 0]) == 2  # counts of the newest table
    assert averaged.meta["rounds_trained"] == 20
    assert averaged.meta["averaged_from"] == [str(p) for p in paths]


def test_averaging_needs_a_table() -> None:
    with pytest.raises(ValueError):
        average_tables([], E1)


@pytest.fixture(scope="module")
def short_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    course = Curriculum(
        name="average-test",
        params={"encoding": "E1"},
        chunk_rounds=1,
        eval_every=1,
        evaluations=(),
        stages=(
            Stage(
                name="coins",
                scenario="coin-heaven",
                lineups=(Lineup(()),),
                rounds=3,
                epsilon_start=0.3,
                epsilon_end=0.3,
            ),
        ),
    )
    run_dir = tmp_path_factory.mktemp("average") / "run_0"
    train_run(course, run_dir, 0)
    return run_dir


def test_a_run_averages_its_newest_snapshots(short_run: Path) -> None:
    out = average_run(short_run, last=2)

    assert out == short_run / "averaged" / "last2_round_000003.npz"
    newest = [QTable.load(path, E1) for _, path in snapshots(short_run)[-2:]]
    averaged = QTable.load(out, E1)
    assert np.allclose(averaged.q, (newest[0].q + newest[1].q) / 2)
    assert np.array_equal(averaged.n, newest[1].n)


def test_a_run_needs_enough_snapshots(short_run: Path) -> None:
    with pytest.raises(ValueError):
        average_run(short_run, last=4)
    with pytest.raises(ValueError):
        average_run(short_run, last=0)


def test_the_cli_averages_every_run_below_a_directory(
    short_run: Path, tmp_path: Path
) -> None:
    assert cli.main(["average", str(short_run.parent), "--last", "3"]) == 0
    assert (short_run / "averaged" / "last3_round_000003.npz").exists()
    assert cli.main(["average", str(tmp_path / "nothing")]) == 2
