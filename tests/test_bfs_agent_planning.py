"""Tests for ``bfs_agent``'s target planning and parameters."""

import json
import logging
import random
from collections.abc import Sequence
from itertools import permutations
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from agent_code.bfs_agent import callbacks
from agent_code.bfs_agent.params import ENV_VAR, Params
from agent_code.bfs_agent.planning import (
    UNREACHABLE,
    CoinRoute,
    DistanceCache,
    Target,
    action_values,
    choose,
    coin_targets,
    distances,
    plan_route,
    route_length,
)
from agent_code.bfs_agent.safety import Board
from agent_code.bfs_agent.world_model import (
    LEFT,
    RIGHT,
    WAIT,
    Geometry,
    Observation,
    Pos,
)
from tests.bfs_boards import arena, game_state, parse_board

# x = 1 .. 9 along y = 1.
LONG = parse_board(["###########", "#.........#", "###########"])


def board_for(
    field: NDArray[np.int64],
    me: Pos,
    *,
    others: Sequence[Pos] = (),
    coins: Sequence[Pos] = (),
    bombs: Sequence[tuple[Pos, int]] = (),
) -> tuple[Observation, Board]:
    obs = Observation.from_game_state(
        game_state(field, me, others=others, coins=coins, bombs=bombs)
    )
    return obs, Board(obs, Geometry(obs.field))


def cache_for(board: Board) -> DistanceCache:
    cache = DistanceCache()
    cache.sync(board)
    return cache


def test_distances_walk_around_crates_and_bombs() -> None:
    field = parse_board(["#######", "#..c..#", "#.###.#", "#.....#", "#######"])
    _, board = board_for(field, (1, 1), bombs=[((1, 3), 2)])

    dist = distances(board, [(1, 1)])

    assert dist[(2, 1)] == 1
    assert (3, 1) not in dist  # the crate
    assert (1, 3) not in dist  # the bomb, which also cuts the only way round
    assert (5, 1) not in dist


def test_distances_from_several_sources_take_the_nearest() -> None:
    _, board = board_for(LONG, (5, 1))
    dist = distances(board, [(1, 1), (9, 1)])
    assert dist[(5, 1)] == 4
    assert dist[(8, 1)] == 1


def test_a_coin_a_rival_reaches_first_is_discounted() -> None:
    obs, board = board_for(LONG, (1, 1), others=[(8, 1)], coins=[(2, 1), (7, 1)])
    params = Params(contested_coin=0.25, off_route=1.0)

    targets = coin_targets(obs, board, cache_for(board), params, CoinRoute())

    assert {t.pos: t.value for t in targets} == {(2, 1): 1.0, (7, 1): 0.25}


def test_unreachable_coins_are_not_targets() -> None:
    field = parse_board(["#######", "#..c..#", "#######"])
    obs, board = board_for(field, (1, 1), coins=[(5, 1)])
    assert coin_targets(obs, board, cache_for(board), Params(), CoinRoute()) == []


def test_moving_towards_a_coin_beats_waiting() -> None:
    obs, board = board_for(LONG, (4, 1), coins=[(7, 1)])
    values = action_values(
        obs, [LEFT, RIGHT, WAIT], [Target((7, 1), 1.0)], cache_for(board), Params()
    )
    assert values[RIGHT] > values[WAIT] > values[LEFT]


def test_route_beats_nearest_first_when_backtracking_is_needed() -> None:
    """From x=5, nearest-first goes 4, 1, 7 (10 steps); 7, 4, 1 takes 8."""
    _, board = board_for(LONG, (5, 1))
    cache = cache_for(board)
    stops = [(4, 1), (1, 1), (7, 1)]

    route = plan_route((5, 1), stops, cache)

    assert route == [(7, 1), (4, 1), (1, 1)]
    assert route_length((5, 1), route, cache) == 8


def test_route_is_never_worse_than_the_best_of_small_random_instances() -> None:
    """On small instances the heuristic route should match brute force."""
    field = arena()
    rng = random.Random(1)
    free = [(x, y) for x in range(1, 16) for y in range(1, 16) if field[x, y] == 0]
    worse = 0
    for _ in range(30):
        start, *stops = rng.sample(free, 6)
        _, board = board_for(field, start)
        cache = cache_for(board)
        best = min(route_length(start, order, cache) for order in permutations(stops))
        planned = route_length(start, plan_route(start, stops, cache), cache)
        assert planned >= best
        worse += planned > best
    assert worse <= 3  # a local search, so allow a rare miss


def test_the_next_coin_on_the_route_gets_full_value() -> None:
    obs, board = board_for(LONG, (5, 1), coins=[(4, 1), (1, 1), (7, 1)])
    targets = coin_targets(
        obs, board, cache_for(board), Params(off_route=0.5), CoinRoute()
    )
    assert {t.pos: t.value for t in targets} == {
        (7, 1): 1.0,
        (4, 1): 0.5,
        (1, 1): 0.5,
    }


def test_the_route_keeps_its_coin_when_another_looks_closer() -> None:
    """Replanning every step made the agent oscillate between two routes."""
    _, board = board_for(LONG, (5, 1))
    cache = cache_for(board)
    route = CoinRoute()
    coins = [(1, 1), (9, 1)]

    first = route.next_coin((5, 1), coins, cache)
    # One step towards the other coin: a fresh plan would switch to it.
    fresh = plan_route((6, 1) if first == (1, 1) else (4, 1), coins, cache)[0]
    kept = route.next_coin((6, 1) if first == (1, 1) else (4, 1), coins, cache)

    assert fresh != first
    assert kept == first


def test_the_route_replans_when_its_coin_is_gone_or_a_new_one_appears() -> None:
    _, board = board_for(LONG, (5, 1))
    cache = cache_for(board)
    route = CoinRoute()

    assert route.next_coin((5, 1), [(1, 1)], cache) == (1, 1)
    assert route.next_coin((5, 1), [(1, 1), (6, 1)], cache) == (6, 1)  # new coin
    assert route.next_coin((6, 1), [(1, 1)], cache) == (1, 1)  # collected


def test_distance_cache_is_dropped_when_a_crate_disappears() -> None:
    field = parse_board(["#######", "#..c..#", "#######"])
    _, board = board_for(field, (1, 1))
    cache = cache_for(board)
    assert cache.between((1, 1), (5, 1)) == UNREACHABLE

    field[3, 1] = 0
    _, opened = board_for(field, (1, 1))
    cache.sync(opened)
    assert cache.between((1, 1), (5, 1)) == 4


def test_choose_breaks_ties_at_random_but_only_among_the_best() -> None:
    rng = random.Random(0)
    values = {LEFT: 1.0, RIGHT: 1.0, WAIT: 0.5}
    picks = {choose(values, rng) for _ in range(100)}
    assert picks == {LEFT, RIGHT}


def test_params_default_without_the_environment_variable() -> None:
    assert Params.from_env({}) == Params()
    assert Params.from_env({ENV_VAR: "  "}) == Params()


def test_params_override_from_the_environment_variable() -> None:
    params = Params.from_env({ENV_VAR: json.dumps({"discount": 0.5, "seed": 3})})
    assert params.discount == 0.5
    assert params.seed == 3
    assert params.coin_value == Params().coin_value


@pytest.mark.parametrize("raw", ['{"no_such_weight": 1}', "[1, 2]"])
def test_params_reject_bad_overrides(raw: str) -> None:
    with pytest.raises(ValueError):
        Params.from_env({ENV_VAR: raw})


def test_the_agent_walks_to_the_nearest_coin() -> None:
    agent = cast(
        callbacks.AgentSelf,
        SimpleNamespace(logger=logging.getLogger("bfs_agent_test"), train=False),
    )
    callbacks.setup(agent)
    state = game_state(arena(), (1, 1), coins=[(4, 1), (1, 9)])
    assert callbacks.act(agent, state) == RIGHT
