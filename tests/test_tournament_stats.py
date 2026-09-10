"""Tests for the tournament statistics."""

import math
from dataclasses import replace

import numpy as np
import pytest

from tournament.results import AgentRoundResult, RoundResult
from tournament.schedule import CANDIDATE_ARM, CONTROL_ARM
from tournament.stats import (
    bootstrap_mean_ci,
    estimate,
    outcome_for,
    paired_delta,
    rounds_for_detectable_difference,
    summarise_arm,
)


def make_agent(
    code_name: str, seat: int, score: int, **overrides: object
) -> AgentRoundResult:
    defaults: dict[str, object] = {
        "code_name": code_name,
        "name": code_name,
        "seat": seat,
        "score": score,
        "coins": score,
        "kills": 0,
        "suicides": 0,
        "crates": 0,
        "bombs": 0,
        "invalid": 0,
        "moves": 10,
        "steps": 10,
        "survived": True,
        "latency_mean": 0.001,
        "latency_p99": 0.002,
        "latency_max": 0.003,
        "timeouts": 0,
    }
    defaults.update(overrides)
    return AgentRoundResult(**defaults)  # pyright: ignore[reportArgumentType]


def make_round(
    scores: list[int],
    *,
    seed: int = 0,
    arm: str = CANDIDATE_ARM,
    focus_seat: int = 0,
    focus_name: str = "candidate",
) -> RoundResult:
    names = [
        focus_name if seat == focus_seat else "rival" for seat in range(len(scores))
    ]
    return RoundResult(
        scenario="classic",
        seed=seed,
        lineup=tuple(names),
        arm=arm,
        focus_seat=focus_seat,
        steps=100,
        agents=tuple(
            make_agent(name, seat, score)
            for seat, (name, score) in enumerate(zip(names, scores, strict=True))
        ),
    )


# --- interval estimation ---------------------------------------------------


def test_bootstrap_recovers_a_known_mean() -> None:
    rng = np.random.default_rng(0)
    sample = [float(x) for x in rng.normal(loc=5.0, scale=2.0, size=800)]

    ci = bootstrap_mean_ci(sample)

    assert ci.low < 5.0 < ci.high
    # A sample that large should pin the mean down tightly.
    assert ci.high - ci.low < 0.6


def test_bootstrap_is_reproducible() -> None:
    sample = [1.0, 4.0, 2.0, 8.0, 3.0]

    assert bootstrap_mean_ci(sample) == bootstrap_mean_ci(sample)


def test_bootstrap_of_a_single_value_is_a_point() -> None:
    assert bootstrap_mean_ci([3.0]) == bootstrap_mean_ci([3.0])
    assert bootstrap_mean_ci([3.0]).low == 3.0
    assert bootstrap_mean_ci([3.0]).high == 3.0


def test_bootstrap_of_nothing_is_undefined() -> None:
    ci = bootstrap_mean_ci([])

    assert math.isnan(ci.low) and math.isnan(ci.high)


def test_estimate_reports_sample_statistics() -> None:
    result = estimate([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], resamples=500)

    assert result.n == 8
    assert result.mean == pytest.approx(5.0)
    # Sample sd (ddof=1), not the population sd of 2.0.
    assert result.sd == pytest.approx(2.13809, abs=1e-4)
    assert result.sem == pytest.approx(result.sd / math.sqrt(8))


def test_estimate_of_one_value_has_no_spread() -> None:
    result = estimate([4.0], resamples=100)

    assert result.mean == 4.0
    assert math.isnan(result.sd)
    assert math.isnan(result.sem)


def test_interval_excludes_zero() -> None:
    assert estimate([1.0] * 20, resamples=200).ci.excludes_zero()
    assert estimate([-1.0] * 20, resamples=200).ci.excludes_zero()
    assert not estimate([-1.0, 1.0] * 20, resamples=200).ci.excludes_zero()


# --- win / tie accounting --------------------------------------------------


def test_outright_top_score_is_a_win() -> None:
    outcome = outcome_for(make_round([7, 3, 2, 1]))

    assert outcome.won and not outcome.tied


def test_shared_top_score_is_a_tie_not_a_win() -> None:
    outcome = outcome_for(make_round([7, 7, 2, 1]))

    assert outcome.tied and not outcome.won


def test_losing_is_neither() -> None:
    outcome = outcome_for(make_round([1, 7, 2, 3]))

    assert not outcome.won and not outcome.tied


def test_win_is_judged_from_the_focus_seat() -> None:
    outcome = outcome_for(make_round([1, 7, 2, 3], focus_seat=1))

    assert outcome.won


# --- arm summaries ---------------------------------------------------------


def test_summary_aggregates_the_focus_agent_only() -> None:
    results = [make_round([4, 9, 9, 9], seed=s) for s in range(5)]

    summary = summarise_arm(results, CANDIDATE_ARM, resamples=200)

    assert summary.rounds == 5
    assert summary.code_name == "candidate"
    assert summary.score.mean == pytest.approx(4.0)
    assert summary.win_rate == 0.0


def test_summary_weights_latency_by_steps() -> None:
    slow = make_round([1], seed=0)
    slow = replace(
        slow, agents=(make_agent("candidate", 0, 1, latency_mean=0.1, steps=100),)
    )
    fast = make_round([1], seed=1)
    fast = replace(
        fast, agents=(make_agent("candidate", 0, 1, latency_mean=0.0, steps=900),)
    )

    summary = summarise_arm([slow, fast], CANDIDATE_ARM, resamples=100)

    # 100 steps at 0.1 s and 900 at 0 s average to 0.01 s, not 0.05 s.
    assert summary.latency_mean == pytest.approx(0.01)


def test_summary_reports_the_true_latency_maximum() -> None:
    calm = replace(
        make_round([1], seed=0),
        agents=(make_agent("candidate", 0, 1, latency_max=0.01),),
    )
    spike = replace(
        make_round([1], seed=1),
        agents=(make_agent("candidate", 0, 1, latency_max=0.42),),
    )

    summary = summarise_arm([calm, spike], CANDIDATE_ARM, resamples=100)

    assert summary.latency_max == pytest.approx(0.42)


def test_summary_rejects_an_empty_arm() -> None:
    with pytest.raises(ValueError, match="no rounds"):
        summarise_arm([make_round([1], arm=CANDIDATE_ARM)], CONTROL_ARM)


def test_summary_rejects_a_mixed_arm() -> None:
    results = [
        make_round([1], seed=0, focus_name="a"),
        make_round([1], seed=1, focus_name="b"),
    ]

    with pytest.raises(ValueError, match="mixes agents"):
        summarise_arm(results, CANDIDATE_ARM)


# --- paired comparison -----------------------------------------------------


def _paired(offsets: list[int]) -> list[RoundResult]:
    """One candidate/control pair per seed, differing by a known offset."""
    results: list[RoundResult] = []
    for seed, offset in enumerate(offsets):
        results.append(
            make_round([3 + offset], seed=seed, arm=CANDIDATE_ARM, focus_name="cand")
        )
        results.append(make_round([3], seed=seed, arm=CONTROL_ARM, focus_name="ctrl"))
    return results


def test_paired_delta_recovers_a_known_offset() -> None:
    delta = paired_delta(_paired([2] * 30), resamples=500)

    assert delta.delta.mean == pytest.approx(2.0)
    assert delta.candidate == "cand"
    assert delta.control == "ctrl"
    assert delta.is_significant


def test_paired_delta_of_identical_arms_is_zero() -> None:
    delta = paired_delta(_paired([0] * 30), resamples=500)

    assert delta.delta.mean == pytest.approx(0.0)
    assert not delta.is_significant


def test_paired_delta_matches_on_board_and_seat() -> None:
    """A control round for a different seat must not be paired to it."""
    results = [
        make_round([5], seed=0, arm=CANDIDATE_ARM, focus_seat=0),
        make_round([1], seed=0, arm=CONTROL_ARM, focus_seat=0),
        make_round([9], seed=0, arm=CANDIDATE_ARM, focus_seat=0),
    ]
    # The third round has no partner and must be dropped, not averaged in.
    delta = paired_delta(results, resamples=100)

    assert delta.delta.n == 1
    assert delta.delta.mean == pytest.approx(4.0)
    assert delta.unpaired_candidate == 1
    assert delta.unpaired_control == 0


def test_paired_delta_needs_both_arms() -> None:
    with pytest.raises(ValueError, match="both a candidate and a control"):
        paired_delta([make_round([1], arm=CANDIDATE_ARM)])


def test_paired_delta_needs_a_shared_board() -> None:
    results = [
        make_round([5], seed=0, arm=CANDIDATE_ARM),
        make_round([1], seed=99, arm=CONTROL_ARM),
    ]

    with pytest.raises(ValueError, match="shares a board"):
        paired_delta(results)


# --- power planning --------------------------------------------------------


def test_power_scales_with_the_square_of_the_effect() -> None:
    coarse = rounds_for_detectable_difference(2.9, 1.0)
    fine = rounds_for_detectable_difference(2.9, 0.5)

    assert fine == pytest.approx(4 * coarse, rel=0.02)


def test_power_matches_the_measured_baseline() -> None:
    # sd 2.88 was measured over 240 agent-rounds of 4x rule_based_agent.
    # 2 * (1.96 * 2.88 / 0.5)^2 = 255 rounds per arm, before pairing helps.
    assert rounds_for_detectable_difference(2.88, 0.5) == 255


def test_power_rejects_nonsense() -> None:
    with pytest.raises(ValueError, match="must both be positive"):
        rounds_for_detectable_difference(0.0, 1.0)
