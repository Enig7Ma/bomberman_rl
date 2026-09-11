"""Tests for latency instrumentation and result storage."""

from dataclasses import replace
from pathlib import Path

import pytest

from tournament.engine import WorldConfig, reset_framework_logging, run_round
from tournament.latency import LatencyRecord, instrument
from tournament.results import AgentRoundResult, RoundResult
from tournament.storage import read_rounds, write_rounds


@pytest.fixture(autouse=True)
def _clean_logging() -> None:
    reset_framework_logging()


class FakeAgent:
    """Stands in for a framework agent with a scripted think-time sequence."""

    def __init__(self, think_times: list[float], budget: float) -> None:
        self._think_times = think_times
        self.available_think_time: float | None = budget

    def wait_for_act(self) -> tuple[str, float]:
        return "WAIT", self._think_times.pop(0)


def test_percentile_uses_nearest_rank() -> None:
    record = LatencyRecord(samples=[float(n) for n in range(1, 101)])

    assert record.percentile(99.0) == 99.0
    assert record.percentile(100.0) == 100.0
    assert record.percentile(50.0) == 50.0
    assert record.maximum == 100.0


def test_empty_record_reports_zeroes() -> None:
    record = LatencyRecord()

    assert record.mean == 0.0
    assert record.maximum == 0.0
    assert record.percentile(99.0) == 0.0


def test_instrument_records_every_call() -> None:
    agent = FakeAgent([0.01, 0.02, 0.03], budget=0.5)
    record = instrument(agent)

    for _ in range(3):
        agent.wait_for_act()

    assert record.samples == [0.01, 0.02, 0.03]
    assert record.timeouts == 0
    assert record.maximum == 0.03


def test_instrument_counts_overruns_against_the_remaining_budget() -> None:
    """Mirrors ``poll_and_run_agents``: the budget shrinks after an overrun."""
    agent = FakeAgent([0.6, 0.3], budget=0.5)
    record = instrument(agent)

    agent.wait_for_act()
    agent.available_think_time = 0.5 - (0.6 - 0.5)  # what the engine would set
    agent.wait_for_act()

    assert record.timeouts == 1


def test_instrument_returns_the_wrapped_action() -> None:
    agent = FakeAgent([0.01], budget=0.5)
    instrument(agent)

    assert agent.wait_for_act() == ("WAIT", 0.01)


def test_round_reports_latency_for_every_agent() -> None:
    result = run_round(
        WorldConfig(lineup=("peaceful_agent", "random_agent"), scenario="empty", seed=2)
    )

    for agent in result.agents:
        assert agent.latency_max >= agent.latency_p99 >= 0.0
        assert agent.latency_max >= agent.latency_mean >= 0.0
        assert agent.timeouts == 0
        # The supplied agents are far inside the 0.5 s budget.
        assert agent.latency_max < 0.5


def _agent_result(code_name: str, seat: int) -> AgentRoundResult:
    return AgentRoundResult(
        code_name=code_name,
        name=code_name,
        seat=seat,
        score=3,
        coins=3,
        kills=0,
        suicides=1,
        crates=4,
        bombs=2,
        invalid=0,
        moves=17,
        steps=20,
        survived=False,
        latency_mean=0.001,
        latency_p99=0.004,
        latency_max=0.005,
        timeouts=0,
    )


def _example_round() -> RoundResult:
    lineup = ("rule_based_agent", "peaceful_agent")
    return RoundResult(
        scenario="classic",
        seed=5,
        lineup=lineup,
        arm="control",
        focus_seat=1,
        steps=20,
        agents=tuple(
            _agent_result(code_name, seat) for seat, code_name in enumerate(lineup)
        ),
    )


def test_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "rounds.jsonl"
    original = _example_round()

    write_rounds(path, [original])

    assert read_rounds(path) == [original]


def test_jsonl_append_accumulates(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "rounds.jsonl"
    original = _example_round()

    write_rounds(path, [original])
    write_rounds(path, [original], append=True)

    assert len(read_rounds(path)) == 2


def test_jsonl_overwrite_replaces(tmp_path: Path) -> None:
    path = tmp_path / "rounds.jsonl"

    write_rounds(path, [_example_round(), _example_round()])
    write_rounds(path, [_example_round()])

    assert len(read_rounds(path)) == 1


def test_read_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "rounds.jsonl"
    write_rounds(path, [_example_round()])
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n\n")

    assert len(read_rounds(path)) == 1


def test_focus_survives_the_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "rounds.jsonl"
    write_rounds(path, [_example_round()])

    assert read_rounds(path)[0].focus.code_name == "peaceful_agent"


def test_focus_rejects_a_seat_that_did_not_play() -> None:
    round_result = _example_round()
    broken = replace(round_result, focus_seat=3)

    with pytest.raises(ValueError, match="seat 3"):
        _ = broken.focus
