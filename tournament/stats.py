"""Turning round records into comparisons with stated uncertainty.

Scores in this game are heavy-tailed and layout-dominated: the measured spread
of ``rule_based_agent`` is a standard deviation of ~2.9 around a mean of ~3.3,
so a point estimate from a handful of rounds says almost nothing. Every number
this module reports therefore comes with an interval, and the headline
comparison is a *paired* difference on shared seeds rather than a difference of
two independently sampled means.

The intervals are percentile bootstrap intervals over rounds. That treats
rounds as the sampling unit, which is right for comparing two fixed policies.
It would *not* be right for comparing two training procedures -- there the
sampling unit is the training seed (see dev/survey.md §7 Phase E).
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from tournament.results import AgentRoundResult, RoundResult
from tournament.schedule import CANDIDATE_ARM, CONTROL_ARM

DEFAULT_RESAMPLES = 10_000
DEFAULT_CONFIDENCE = 0.95
BOOTSTRAP_SEED = 20260910

# Pairing key: a candidate round and its control round describe the same board
# played from the same corner.
PairKey = tuple[str, int, int]


@dataclass(frozen=True)
class Interval:
    """A confidence interval."""

    low: float
    high: float

    def excludes_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0


@dataclass(frozen=True)
class Estimate:
    """A mean with its spread and a bootstrap interval."""

    n: int
    mean: float
    sd: float
    sem: float
    ci: Interval


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Interval:
    """Percentile bootstrap interval for the mean.

    Fixed seed by default so a report is reproducible from its JSONL.
    """
    if not values:
        return Interval(math.nan, math.nan)
    if len(values) == 1:
        only = float(values[0])
        return Interval(only, only)

    observations = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = observations.size

    # Drawn in chunks so a large resample count stays memory-bounded.
    means: list[np.ndarray] = []
    remaining = resamples
    while remaining > 0:
        block = min(remaining, 500)
        picks = rng.integers(0, n, size=(block, n))
        means.append(observations[picks].mean(axis=1))
        remaining -= block

    distribution = np.concatenate(means)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(distribution, [tail, 1.0 - tail])
    return Interval(float(low), float(high))


def estimate(
    values: Sequence[float],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Estimate:
    """Summarise a sample: mean, sd, sem and a bootstrap interval."""
    n = len(values)
    if n == 0:
        return Estimate(0, math.nan, math.nan, math.nan, Interval(math.nan, math.nan))

    observations = np.asarray(values, dtype=np.float64)
    mean = float(observations.mean())
    # Sample standard deviation; undefined for a single observation.
    sd = float(observations.std(ddof=1)) if n > 1 else math.nan
    sem = sd / math.sqrt(n) if n > 1 else math.nan
    ci = bootstrap_mean_ci(
        values, confidence=confidence, resamples=resamples, seed=seed
    )
    return Estimate(n=n, mean=mean, sd=sd, sem=sem, ci=ci)


@dataclass(frozen=True)
class Outcome:
    """How a round ended for the agent under measurement."""

    won: bool
    tied: bool


def outcome_for(result: RoundResult) -> Outcome:
    """Did the focus agent take the round outright, share the top, or lose?

    A tie is recorded as a tie, never as a win. The GUI breaks score ties by
    name; that is a display artefact and must not become a metric.
    """
    focus = result.focus
    best = max(agent.score for agent in result.agents)
    if focus.score < best:
        return Outcome(won=False, tied=False)
    sharers = sum(1 for agent in result.agents if agent.score == best)
    return Outcome(won=sharers == 1, tied=sharers > 1)


@dataclass(frozen=True)
class ArmSummary:
    """Everything reported for one arm of a comparison."""

    arm: str
    code_name: str
    rounds: int
    score: Estimate
    coins: float
    kills: float
    suicides: float
    crates: float
    survival_rate: float
    win_rate: float
    tie_rate: float
    invalid: float
    latency_mean: float
    latency_p99_worst: float
    latency_max: float
    timeouts: int


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else math.nan


def summarise_arm(
    results: Sequence[RoundResult],
    arm: str,
    *,
    resamples: int = DEFAULT_RESAMPLES,
) -> ArmSummary:
    """Aggregate the focus agent's rounds for one arm.

    ``latency_p99_worst`` is the worst *round's* p99, not a p99 pooled over all
    steps: per-round records do not carry the individual samples. Use
    ``latency_max``, which is a true maximum, to check the timeout budget.
    """
    rounds = [result for result in results if result.arm == arm]
    if not rounds:
        raise ValueError(f"no rounds recorded for arm {arm!r}")

    focus: list[AgentRoundResult] = [result.focus for result in rounds]
    outcomes = [outcome_for(result) for result in rounds]
    names = {agent.code_name for agent in focus}
    if len(names) > 1:
        raise ValueError(f"arm {arm!r} mixes agents: {sorted(names)}")

    steps = sum(agent.steps for agent in focus)
    weighted_latency = sum(agent.latency_mean * agent.steps for agent in focus)

    return ArmSummary(
        arm=arm,
        code_name=names.pop(),
        rounds=len(rounds),
        score=estimate([float(agent.score) for agent in focus], resamples=resamples),
        coins=_mean(float(agent.coins) for agent in focus),
        kills=_mean(float(agent.kills) for agent in focus),
        suicides=_mean(float(agent.suicides) for agent in focus),
        crates=_mean(float(agent.crates) for agent in focus),
        survival_rate=_mean(float(agent.survived) for agent in focus),
        win_rate=_mean(float(outcome.won) for outcome in outcomes),
        tie_rate=_mean(float(outcome.tied) for outcome in outcomes),
        invalid=_mean(float(agent.invalid) for agent in focus),
        latency_mean=weighted_latency / steps if steps else math.nan,
        latency_p99_worst=max((agent.latency_p99 for agent in focus), default=math.nan),
        latency_max=max((agent.latency_max for agent in focus), default=math.nan),
        timeouts=sum(agent.timeouts for agent in focus),
    )


@dataclass(frozen=True)
class PairedDelta:
    """Candidate minus control, on rounds that share a board and a seat."""

    candidate: str
    control: str
    delta: Estimate
    unpaired_candidate: int
    unpaired_control: int

    @property
    def is_significant(self) -> bool:
        """Whether the interval excludes zero. Not a claim about size."""
        return self.delta.ci.excludes_zero()


def _pair_key(result: RoundResult) -> PairKey:
    return (result.scenario, result.seed, result.focus_seat)


def paired_delta(
    results: Sequence[RoundResult],
    *,
    resamples: int = DEFAULT_RESAMPLES,
) -> PairedDelta:
    """Difference in focus score between the two arms, paired on the board.

    Rounds without a partner in the other arm are counted and dropped rather
    than silently mixed into an unpaired comparison.
    """
    candidates: dict[PairKey, list[RoundResult]] = {}
    controls: dict[PairKey, list[RoundResult]] = {}
    for result in results:
        target = (
            candidates
            if result.arm == CANDIDATE_ARM
            else controls
            if result.arm == CONTROL_ARM
            else None
        )
        if target is not None:
            target.setdefault(_pair_key(result), []).append(result)

    if not candidates or not controls:
        raise ValueError("paired comparison needs both a candidate and a control arm")

    deltas: list[float] = []
    unpaired_candidate = 0
    unpaired_control = 0
    for key in candidates.keys() | controls.keys():
        mine = candidates.get(key, [])
        theirs = controls.get(key, [])
        for candidate_round, control_round in zip(mine, theirs, strict=False):
            deltas.append(
                float(candidate_round.focus.score - control_round.focus.score)
            )
        unpaired_candidate += max(0, len(mine) - len(theirs))
        unpaired_control += max(0, len(theirs) - len(mine))

    if not deltas:
        raise ValueError("no candidate round shares a board with a control round")

    candidate_names = {r.focus.code_name for rs in candidates.values() for r in rs}
    control_names = {r.focus.code_name for rs in controls.values() for r in rs}
    return PairedDelta(
        candidate="/".join(sorted(candidate_names)),
        control="/".join(sorted(control_names)),
        delta=estimate(deltas, resamples=resamples),
        unpaired_candidate=unpaired_candidate,
        unpaired_control=unpaired_control,
    )


def rounds_for_detectable_difference(sd: float, difference: float) -> int:
    """Rounds per arm needed to resolve ``difference`` at 95% confidence.

    The usual two-sample normal approximation, ``n = 2 (1.96 sd / d)^2``. It is
    a planning aid, not a guarantee: it ignores the pairing that shrinks the
    real requirement and the non-normality that inflates it.
    """
    if sd <= 0.0 or difference <= 0.0:
        raise ValueError("sd and difference must both be positive")
    return math.ceil(2.0 * (1.96 * sd / difference) ** 2)
