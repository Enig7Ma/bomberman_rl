"""Tests for the per-round world runner."""

import logging

import numpy as np
import pytest

import settings
from tournament.engine import (
    WorldConfig,
    create_world,
    quiet_logging,
    reset_framework_logging,
    run_round,
)
from tournament.framework import agents_of

# A single agent on an empty board finishes in one step: there is nothing left
# to do, so ``time_to_stop`` fires immediately. Keeps the tests fast.
SOLO_EMPTY = WorldConfig(lineup=("peaceful_agent",), scenario="empty", seed=1)


@pytest.fixture(autouse=True)
def _clean_logging() -> None:
    reset_framework_logging()


def test_round_completes_and_reports_every_seat() -> None:
    config = WorldConfig(
        lineup=("peaceful_agent", "random_agent"), scenario="empty", seed=3
    )
    result = run_round(config)

    assert result.steps >= 1
    assert result.scenario == "empty"
    assert result.seed == 3
    assert [agent.code_name for agent in result.agents] == [
        "peaceful_agent",
        "random_agent",
    ]
    assert [agent.seat for agent in result.agents] == [0, 1]


def test_focus_selects_the_scheduled_seat() -> None:
    config = WorldConfig(
        lineup=("peaceful_agent", "random_agent"),
        scenario="empty",
        seed=3,
        focus_seat=1,
    )
    result = run_round(config)

    assert result.focus.code_name == "random_agent"
    assert result.focus.seat == 1


def test_duplicate_agents_are_reported_under_distinct_names() -> None:
    config = WorldConfig(lineup=("peaceful_agent",) * 2, scenario="empty", seed=4)
    result = run_round(config)

    assert [agent.name for agent in result.agents] == [
        "peaceful_agent_0",
        "peaceful_agent_1",
    ]
    assert {agent.code_name for agent in result.agents} == {"peaceful_agent"}


def test_log_handlers_do_not_accumulate_across_rounds() -> None:
    for seed in range(3):
        run_round(WorldConfig(lineup=("peaceful_agent",), scenario="empty", seed=seed))

    assert logging.getLogger("BombeRLeWorld").handlers == []
    assert logging.getLogger("peaceful_agent_code").handlers == []
    assert logging.getLogger("peaceful_agent_wrapper").handlers == []


def test_same_seed_gives_the_same_arena_for_different_lineups() -> None:
    """The pairing property the whole comparison design rests on.

    A candidate and its control must face the same arena from the same corners,
    which only holds because each round gets a freshly seeded world.
    """
    arenas: list[np.ndarray] = []
    seats: list[list[tuple[int, int]]] = []
    for lineup in (("peaceful_agent",) * 4, ("random_agent",) * 4):
        world = create_world(WorldConfig(lineup=lineup, scenario="classic", seed=11))
        try:
            world.new_round()
            arenas.append(np.array(world.arena))
            seats.append([(a.x, a.y) for a in agents_of(world)])
        finally:
            reset_framework_logging()

    assert np.array_equal(arenas[0], arenas[1])
    assert seats[0] == seats[1]


def test_arena_does_not_depend_on_earlier_rounds() -> None:
    """Fresh worlds keep seeds stable; a reused world would not.

    ``BombeRLeWorld`` shares one RNG between arena generation and the per-step
    action-order permutation, so stepping one world through several rounds makes
    round N's arena depend on how long rounds 1..N-1 took.
    """

    def arena_for_seed(seed: int) -> np.ndarray:
        world = create_world(
            WorldConfig(lineup=("random_agent",), scenario="classic", seed=seed)
        )
        try:
            world.new_round()
            return np.array(world.arena)
        finally:
            reset_framework_logging()

    first = arena_for_seed(7)
    run_round(WorldConfig(lineup=("random_agent",), scenario="classic", seed=99))
    second = arena_for_seed(7)

    assert np.array_equal(first, second)


def test_quiet_logging_restores_levels() -> None:
    before = (settings.LOG_GAME, settings.LOG_AGENT_WRAPPER, settings.LOG_AGENT_CODE)

    with quiet_logging():
        assert settings.LOG_GAME > logging.CRITICAL
        assert settings.LOG_AGENT_CODE > logging.CRITICAL

    assert (
        settings.LOG_GAME,
        settings.LOG_AGENT_WRAPPER,
        settings.LOG_AGENT_CODE,
    ) == before


def test_reset_framework_logging_is_idempotent() -> None:
    run_round(SOLO_EMPTY)
    reset_framework_logging()
    reset_framework_logging()

    assert logging.getLogger("BombeRLeWorld").handlers == []
