"""Building the set of rounds a comparison needs.

Two design choices carry the statistics:

**Seat rotation.** Start corners are assigned by the world RNG, so over many
rounds they balance out in expectation. Emitting every rotation of the same
seed set balances them exactly instead, which removes corner luck from the
comparison rather than averaging over it.

**Paired controls.** For a candidate lineup, the same schedule is also emitted
with the incumbent in the candidate's seat, on the same seeds. Because the
arena and seating are a function of the seed alone (see ``engine``), the
candidate and its control face the same board from the same corner, and the
per-seed difference cancels the layout variance that dominates raw scores.

What pairing does *not* remove is opponent behaviour: ``rule_based_agent``
seeds itself from entropy and uses the global RNG, so the opponents' dice
differ between the two arms. Repetition, not seeding, handles that.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from tournament.engine import WorldConfig

CANDIDATE_ARM = "candidate"
CONTROL_ARM = "control"


@dataclass(frozen=True)
class Preset:
    """A named comparison: who the candidate plays, where, and how often."""

    opponents: tuple[str, ...]
    scenario: str
    control: str | None = None
    description: str = ""

    @property
    def seats(self) -> int:
        return len(self.opponents) + 1


PRESETS: dict[str, Preset] = {
    "vs-rule-based": Preset(
        opponents=("rule_based_agent",) * 3,
        scenario="classic",
        control="rule_based_agent",
        description="Tournament conditions: the headline comparison.",
    ),
    "mirror": Preset(
        opponents=("rule_based_agent",) * 3,
        scenario="classic",
        control="rule_based_agent",
        description="Null test: candidate is the incumbent, so the delta should be ~0.",
    ),
    "vs-peaceful": Preset(
        opponents=("peaceful_agent",) * 3,
        scenario="classic",
        control="rule_based_agent",
        description="Course task 3 (easy): opponents that never bomb.",
    ),
    "vs-coin-collector": Preset(
        opponents=("coin_collector_agent",) * 3,
        scenario="classic",
        control="rule_based_agent",
        description="Course task 3 (hard): opponents that bomb only for coins.",
    ),
    "coin-heaven-solo": Preset(
        opponents=(),
        scenario="coin-heaven",
        control="rule_based_agent",
        description="Course task 1: navigation only, no crates and no opponents.",
    ),
    "crates-solo": Preset(
        opponents=(),
        scenario="loot-crate",
        control="rule_based_agent",
        description="Course task 2: demolition without opponents, 50 coins.",
    ),
}


def seeds(count: int, start: int = 0) -> tuple[int, ...]:
    """A contiguous seed set. Explicit so tuning and test sets cannot overlap."""
    if count < 1:
        raise ValueError(f"need at least one seed, got {count}")
    return tuple(range(start, start + count))


def lineup_with(candidate: str, opponents: Sequence[str], seat: int) -> tuple[str, ...]:
    """Insert the candidate at ``seat``, keeping the opponents in order."""
    if not 0 <= seat <= len(opponents):
        raise ValueError(f"seat {seat} out of range for {len(opponents)} opponents")
    entries = list(opponents)
    entries.insert(seat, candidate)
    return tuple(entries)


def build_schedule(
    candidate: str,
    preset: Preset,
    seed_set: Sequence[int],
    *,
    with_control: bool = True,
) -> list[WorldConfig]:
    """Every round needed to compare ``candidate`` under ``preset``.

    Rounds are emitted seed-major, then rotation, then arm, so a truncated run
    is still balanced across seats rather than biased towards seat 0.
    """
    if not seed_set:
        raise ValueError("need at least one seed")

    control = preset.control if with_control else None
    schedule: list[WorldConfig] = []
    for seed in seed_set:
        for seat in range(preset.seats):
            schedule.append(
                WorldConfig(
                    lineup=lineup_with(candidate, preset.opponents, seat),
                    scenario=preset.scenario,
                    seed=seed,
                    arm=CANDIDATE_ARM,
                    focus_seat=seat,
                )
            )
            if control is not None:
                schedule.append(
                    WorldConfig(
                        lineup=lineup_with(control, preset.opponents, seat),
                        scenario=preset.scenario,
                        seed=seed,
                        arm=CONTROL_ARM,
                        focus_seat=seat,
                    )
                )
    return schedule
