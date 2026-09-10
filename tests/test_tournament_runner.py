"""Tests for schedule execution."""

from pathlib import Path

import pytest

from tournament.engine import WorldConfig, reset_framework_logging
from tournament.runner import run_schedule
from tournament.schedule import PRESETS, build_schedule, seeds

# Solo agents on an empty board finish in a single step, so these run fast
# enough to exercise the pool without dominating the test suite.
SCHEDULE = [
    WorldConfig(lineup=("peaceful_agent",), scenario="empty", seed=seed)
    for seed in range(6)
]


@pytest.fixture(autouse=True)
def _clean_logging() -> None:
    reset_framework_logging()


def test_empty_schedule_runs_nothing() -> None:
    assert run_schedule([]) == []


def test_rejects_zero_jobs() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        run_schedule(SCHEDULE, jobs=0)


def test_sequential_returns_one_result_per_config() -> None:
    results = run_schedule(SCHEDULE, jobs=1)

    assert len(results) == len(SCHEDULE)
    assert [r.seed for r in results] == [c.seed for c in SCHEDULE]


def test_parallel_preserves_schedule_order() -> None:
    results = run_schedule(SCHEDULE, jobs=3)

    assert [r.seed for r in results] == [c.seed for c in SCHEDULE]


def test_parallel_and_sequential_agree_on_arenas() -> None:
    """Same seeds must give the same games however the work is distributed.

    Step counts are the observable: with a deterministic agent on an empty
    board the round length is a function of the arena alone.
    """
    sequential = run_schedule(SCHEDULE, jobs=1)
    parallel = run_schedule(SCHEDULE, jobs=3)

    assert [r.steps for r in parallel] == [r.steps for r in sequential]
    assert [r.lineup for r in parallel] == [r.lineup for r in sequential]


def test_results_carry_their_schedule_metadata() -> None:
    schedule = build_schedule("peaceful_agent", PRESETS["coin-heaven-solo"], seeds(2))
    results = run_schedule(schedule, jobs=1)

    assert [(r.arm, r.focus_seat, r.seed) for r in results] == [
        (c.arm, c.focus_seat, c.seed) for c in schedule
    ]


def test_progress_bar_does_not_change_results() -> None:
    quiet = run_schedule(SCHEDULE[:2], jobs=1)
    noisy = run_schedule(SCHEDULE[:2], jobs=1, progress=True)

    assert [r.seed for r in quiet] == [r.seed for r in noisy]


def test_sequential_creates_its_log_directory(tmp_path: Path) -> None:
    log_dir = tmp_path / "made-up"
    run_schedule(SCHEDULE[:1], jobs=1, log_dir=str(log_dir))

    assert (log_dir / "game.log").exists()
