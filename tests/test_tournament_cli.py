"""End-to-end tests for the CLI and the markdown report."""

from dataclasses import replace
from pathlib import Path

import pytest

from tests.factories import make_agent, make_round
from tournament.cli import main
from tournament.engine import reset_framework_logging
from tournament.report import render
from tournament.schedule import CANDIDATE_ARM, CONTROL_ARM
from tournament.storage import read_rounds


@pytest.fixture(autouse=True)
def _clean_logging() -> None:
    reset_framework_logging()


def test_run_writes_rounds_and_reports(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "rounds.jsonl"

    exit_code = main(
        [
            "run",
            "--candidate",
            "peaceful_agent",
            "--preset",
            "coin-heaven-solo",
            "--rounds",
            "2",
            "--out",
            str(out),
            "--quiet",
        ]
    )

    assert exit_code == 0
    results = read_rounds(out)
    # 2 seeds x 1 seat x 2 arms.
    assert len(results) == 4
    assert {r.arm for r in results} == {CANDIDATE_ARM, CONTROL_ARM}


def test_run_without_control_halves_the_work(tmp_path: Path) -> None:
    out = tmp_path / "rounds.jsonl"

    main(
        [
            "run",
            "--candidate",
            "peaceful_agent",
            "--preset",
            "coin-heaven-solo",
            "--rounds",
            "2",
            "--out",
            str(out),
            "--no-control",
            "--quiet",
        ]
    )

    assert {r.arm for r in read_rounds(out)} == {CANDIDATE_ARM}


def test_run_append_accumulates(tmp_path: Path) -> None:
    out = tmp_path / "rounds.jsonl"
    argv = [
        "run",
        "--candidate",
        "peaceful_agent",
        "--preset",
        "coin-heaven-solo",
        "--rounds",
        "1",
        "--out",
        str(out),
        "--no-control",
        "--quiet",
    ]

    main(argv)
    main([*argv, "--append"])

    assert len(read_rounds(out)) == 2


def test_seed_start_selects_a_disjoint_set(tmp_path: Path) -> None:
    tuning, holdout = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    base = [
        "run",
        "--candidate",
        "peaceful_agent",
        "--preset",
        "coin-heaven-solo",
        "--rounds",
        "2",
        "--no-control",
        "--quiet",
    ]

    main([*base, "--out", str(tuning)])
    main([*base, "--out", str(holdout), "--seed-start", "500"])

    assert {r.seed for r in read_rounds(tuning)}.isdisjoint(
        {r.seed for r in read_rounds(holdout)}
    )


def test_report_reads_back_what_run_wrote(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "rounds.jsonl"
    main(
        [
            "run",
            "--candidate",
            "peaceful_agent",
            "--preset",
            "coin-heaven-solo",
            "--rounds",
            "2",
            "--out",
            str(out),
            "--quiet",
        ]
    )
    capsys.readouterr()

    assert main(["report", str(out), "--title", "Smoke"]) == 0
    printed = capsys.readouterr().out
    assert "Smoke" in printed
    assert "Paired score difference" in printed
    assert "peaceful_agent" in printed


def test_report_numbers_match_the_records(tmp_path: Path) -> None:
    out = tmp_path / "rounds.jsonl"
    main(
        [
            "run",
            "--candidate",
            "peaceful_agent",
            "--preset",
            "coin-heaven-solo",
            "--rounds",
            "3",
            "--out",
            str(out),
            "--no-control",
            "--quiet",
        ]
    )
    results = read_rounds(out)

    expected = sum(r.focus.score for r in results) / len(results)
    rendered = render(results, budget=0.5)

    assert f"| {expected:.2f} |" in rendered


def test_report_of_nothing_says_so() -> None:
    assert render([], budget=0.5) == "No rounds to report."


def test_report_warns_about_timeouts() -> None:
    slow = replace(
        make_round([1], seed=0),
        agents=(make_agent("slowpoke", 0, 1, timeouts=2, latency_max=0.9),),
    )
    rendered = render([slow], budget=0.5)

    assert "Warnings" in rendered
    assert "exceeded the think-time budget" in rendered


def test_report_warns_when_latency_approaches_the_budget() -> None:
    close = replace(
        make_round([1], seed=0),
        agents=(make_agent("borderline", 0, 1, latency_max=0.3),),
    )
    rendered = render([close], budget=0.5)

    assert "peaked at" in rendered


def test_unknown_preset_is_rejected() -> None:
    with pytest.raises(SystemExit):
        main(["run", "--candidate", "x", "--preset", "nope", "--out", "x.jsonl"])


def test_report_shows_round_length() -> None:
    rendered = render([replace(make_round([1], seed=0), steps=124)], budget=0.5)

    assert "round len" in rendered
    assert "| 124 |" in rendered
