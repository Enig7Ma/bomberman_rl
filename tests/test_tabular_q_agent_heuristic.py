"""Tests for ``tabular_q_agent``'s fixed-priority control policy."""

import random

import pytest

from agent_code.tabular_q_agent.features import (
    DOWN,
    HERE,
    LEFT,
    NO_ATTACK,
    NONE,
    PRESSURE,
    RIGHT,
    TRAP,
    UP,
    Features,
)
from agent_code.tabular_q_agent.heuristic import heuristic_action

ALL = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")


def choose(features: Features, allowed: tuple[str, ...] = ALL, seed: int = 0) -> str:
    return heuristic_action(features, allowed, random.Random(seed))


def test_a_coin_comes_first() -> None:
    features = Features(coin_dir=LEFT, crate_dir=UP, bomb_yield=3, attack=TRAP)
    assert choose(features) == "LEFT"


def test_a_coin_underfoot_is_collected_by_waiting() -> None:
    assert choose(Features(coin_dir=HERE, crate_dir=UP)) == "WAIT"


def test_a_coin_move_the_mask_forbids_is_skipped() -> None:
    features = Features(coin_dir=LEFT, crate_dir=DOWN)
    assert choose(features, ("UP", "DOWN", "WAIT")) == "DOWN"


@pytest.mark.parametrize(
    ("bomb_yield", "attack"), [(1, NO_ATTACK), (0, PRESSURE), (0, TRAP)]
)
def test_bombs_where_a_bomb_is_worth_something(bomb_yield: int, attack: int) -> None:
    features = Features(crate_dir=RIGHT, bomb_yield=bomb_yield, attack=attack)
    assert choose(features) == "BOMB"


def test_no_bomb_when_the_mask_forbids_it() -> None:
    features = Features(crate_dir=HERE, opp_dir=UP, bomb_yield=2)
    assert choose(features, ("UP", "WAIT")) == "UP"


def test_a_crate_spot_before_an_opponent() -> None:
    assert choose(Features(crate_dir=DOWN, opp_dir=UP)) == "DOWN"


def test_an_opponent_when_nothing_else_is_there() -> None:
    assert choose(Features(opp_dir=RIGHT)) == "RIGHT"


def test_otherwise_any_allowed_action() -> None:
    allowed = ("UP", "WAIT")
    picks = {choose(Features(coin_dir=NONE), allowed, seed) for seed in range(30)}
    assert picks == set(allowed)


def test_the_mask_is_never_empty() -> None:
    with pytest.raises(ValueError):
        choose(Features(), ())
