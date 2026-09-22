"""Tests for match scheduling."""

from collections import Counter

import pytest

from tournament.schedule import (
    CANDIDATE_ARM,
    CONTROL_ARM,
    PRESETS,
    Preset,
    build_schedule,
    lineup_with,
    seeds,
)

FOUR_WAY = PRESETS["vs-rule-based"]
SOLO = PRESETS["coin-heaven-solo"]


def test_lineup_places_the_candidate_in_the_requested_seat() -> None:
    opponents = ("a", "b", "c")

    assert lineup_with("me", opponents, 0) == ("me", "a", "b", "c")
    assert lineup_with("me", opponents, 2) == ("a", "b", "me", "c")
    assert lineup_with("me", opponents, 3) == ("a", "b", "c", "me")


def test_lineup_rejects_an_impossible_seat() -> None:
    with pytest.raises(ValueError, match="seat 4"):
        lineup_with("me", ("a", "b", "c"), 4)


def test_seeds_are_contiguous_and_disjoint_when_offset() -> None:
    tuning = seeds(5)
    holdout = seeds(5, start=1000)

    assert tuning == (0, 1, 2, 3, 4)
    assert set(tuning).isdisjoint(holdout)


def test_seeds_rejects_an_empty_set() -> None:
    with pytest.raises(ValueError, match="at least one seed"):
        seeds(0)


def test_every_seat_is_used_equally_often() -> None:
    schedule = build_schedule("bfs_agent", FOUR_WAY, seeds(3))
    candidate = [c for c in schedule if c.arm == CANDIDATE_ARM]

    assert Counter(c.focus_seat for c in candidate) == {0: 3, 1: 3, 2: 3, 3: 3}


def test_the_candidate_actually_occupies_its_focus_seat() -> None:
    schedule = build_schedule("bfs_agent", FOUR_WAY, seeds(2))

    for config in schedule:
        expected = "bfs_agent" if config.arm == CANDIDATE_ARM else "rule_based_agent"
        assert config.lineup[config.focus_seat] == expected


def test_control_mirrors_the_candidate_schedule() -> None:
    schedule = build_schedule("bfs_agent", FOUR_WAY, seeds(3))

    def keys(arm: str) -> Counter[tuple[int, int]]:
        return Counter((c.seed, c.focus_seat) for c in schedule if c.arm == arm)

    assert keys(CANDIDATE_ARM) == keys(CONTROL_ARM)


def test_control_replaces_only_the_candidate() -> None:
    schedule = build_schedule("bfs_agent", FOUR_WAY, seeds(1))
    control = next(c for c in schedule if c.arm == CONTROL_ARM and c.focus_seat == 2)

    assert control.lineup == (
        "rule_based_agent",
        "rule_based_agent",
        "rule_based_agent",
        "rule_based_agent",
    )


def test_control_can_be_suppressed() -> None:
    schedule = build_schedule("bfs_agent", FOUR_WAY, seeds(2), with_control=False)

    assert all(config.arm == CANDIDATE_ARM for config in schedule)
    assert len(schedule) == 2 * FOUR_WAY.seats


def test_solo_preset_has_a_single_seat() -> None:
    schedule = build_schedule("bfs_agent", SOLO, seeds(4))
    candidate = [c for c in schedule if c.arm == CANDIDATE_ARM]

    assert SOLO.seats == 1
    assert all(c.lineup == ("bfs_agent",) for c in candidate)
    assert len(candidate) == 4


def test_schedule_is_deterministic() -> None:
    first = build_schedule("bfs_agent", FOUR_WAY, seeds(3))
    second = build_schedule("bfs_agent", FOUR_WAY, seeds(3))

    assert first == second


def test_truncating_a_run_stays_seat_balanced() -> None:
    """Rounds are emitted seed-major so a partial run is not biased to seat 0."""
    schedule = build_schedule("bfs_agent", FOUR_WAY, seeds(10))
    prefix = [c for c in schedule[: 2 * FOUR_WAY.seats] if c.arm == CANDIDATE_ARM]

    assert sorted(c.focus_seat for c in prefix) == [0, 1, 2, 3]


def test_schedule_rejects_an_empty_seed_set() -> None:
    with pytest.raises(ValueError, match="at least one seed"):
        build_schedule("bfs_agent", FOUR_WAY, ())


def test_presets_reference_agents_that_exist() -> None:
    from pathlib import Path

    from tournament.engine import REPO_ROOT

    for name, preset in PRESETS.items():
        referenced = set(preset.opponents) | ({preset.control} - {None})
        for code_name in referenced:
            assert code_name is not None
            directory = Path(REPO_ROOT, "agent_code", code_name)
            assert directory.is_dir(), f"preset {name} references missing {code_name}"


def test_preset_seats_never_exceed_the_board_limit() -> None:
    import settings

    for name, preset in PRESETS.items():
        assert preset.seats <= settings.MAX_AGENTS, name


def test_custom_preset_without_control_emits_candidate_only() -> None:
    preset = Preset(opponents=("peaceful_agent",), scenario="empty")
    schedule = build_schedule("random_agent", preset, seeds(2))

    assert all(config.arm == CANDIDATE_ARM for config in schedule)


def test_head_to_head_preset_pairs_the_two_learned_models() -> None:
    """``tabular-control`` answers "is the network better than the table?" the
    same way ``bfs-control`` asks it of the search agent: tournament
    conditions, and the control arm puts the other learned model in the
    candidate's seat on the same board."""
    preset = PRESETS["tabular-control"]
    assert preset.control == "tabular_q_agent"
    assert preset.opponents == ("rule_based_agent",) * 3
    schedule = build_schedule("dqn_agent", preset, seeds(3, 500))
    candidate = [c for c in schedule if c.arm == CANDIDATE_ARM]
    control = [c for c in schedule if c.arm == CONTROL_ARM]
    assert len(candidate) == len(control) == 3 * preset.seats
    for mine, theirs in zip(candidate, control, strict=True):
        assert (mine.seed, mine.focus_seat) == (theirs.seed, theirs.focus_seat)
        assert mine.lineup[mine.focus_seat] == "dqn_agent"
        assert theirs.lineup[theirs.focus_seat] == "tabular_q_agent"
        assert theirs.lineup.count("tabular_q_agent") == 1
