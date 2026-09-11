"""Tests for ``bfs_agent``'s choice of where to drop bombs on crates."""

import logging
from collections.abc import Sequence
from types import SimpleNamespace
from typing import cast

import numpy as np
from numpy.typing import NDArray

from agent_code.bfs_agent import callbacks
from agent_code.bfs_agent.params import Params
from agent_code.bfs_agent.planning import (
    DistanceCache,
    Target,
    action_values,
    bomb_spots,
    can_flee,
)
from agent_code.bfs_agent.safety import Board
from agent_code.bfs_agent.world_model import (
    BOMB,
    RIGHT,
    WAIT,
    Geometry,
    Observation,
    Pos,
    danger_timeline,
)
from tests.bfs_boards import arena, game_state, parse_board

CRATE_VALUE = 0.3


def spots_for(
    field: NDArray[np.int64],
    me: Pos,
    *,
    bombs: Sequence[tuple[Pos, int]] = (),
) -> dict[Pos, float]:
    obs = Observation.from_game_state(game_state(field, me, bombs=bombs))
    geometry = Geometry(obs.field)
    board = Board(obs, geometry)
    cache = DistanceCache()
    cache.sync(board)
    timeline = danger_timeline(geometry, board.crates, obs.bombs, obs.explosion_map)
    params = Params(crate_value=CRATE_VALUE)
    return {t.pos: t.value for t in bomb_spots(obs, board, cache, timeline, params)}


def make_agent() -> callbacks.AgentSelf:
    agent = cast(
        callbacks.AgentSelf,
        SimpleNamespace(logger=logging.getLogger("bfs_agent_test"), train=False),
    )
    callbacks.setup(agent)
    return agent


def test_a_spot_counts_every_crate_its_blast_passes_through() -> None:
    # From (2, 1) the blast runs right through (3, 1) and on to (5, 1).
    spots = spots_for(arena(crates=[(3, 1), (5, 1)]), (1, 1))

    assert spots[(2, 1)] == 2 * CRATE_VALUE
    assert spots[(1, 1)] == 1 * CRATE_VALUE
    assert (4, 1) not in spots  # behind the crate at (3, 1): unreachable


def test_crates_a_live_bomb_will_destroy_are_not_counted_twice() -> None:
    field = arena(crates=[(3, 1), (5, 1)])
    # The bomb at (5, 3) will take out (5, 1).
    spots = spots_for(field, (1, 1), bombs=[((5, 3), 3)])
    assert spots[(2, 1)] == 1 * CRATE_VALUE


def test_a_dead_end_spot_is_not_a_target_but_one_with_a_side_exit_is() -> None:
    dead_end = parse_board(["#######", "#....c#", "#######"])
    side_exit = parse_board(["#######", "#....c#", "#.#####", "#######"])

    assert spots_for(dead_end, (1, 1)) == {}
    # Every cell within 3 of the crate reaches it, and (1, 2) is a way out.
    assert spots_for(side_exit, (1, 1)) == {
        (2, 1): CRATE_VALUE,
        (3, 1): CRATE_VALUE,
        (4, 1): CRATE_VALUE,
    }


def test_can_flee_needs_a_cell_outside_the_blast_within_the_fuse() -> None:
    field = parse_board(["#######", "#....c#", "#.#####", "#######"])
    obs = Observation.from_game_state(game_state(field, (1, 1)))
    board = Board(obs, Geometry(obs.field))

    assert can_flee(board, (4, 1))  # 3 moves west, then 1 south
    # From the corner, every cell within 4 moves is inside the blast.
    assert not can_flee(board, (1, 1))


def test_bombing_now_beats_walking_to_an_equally_good_spot() -> None:
    field = arena(crates=[(4, 1)])
    obs = Observation.from_game_state(game_state(field, (1, 1)))
    board = Board(obs, Geometry(obs.field))
    cache = DistanceCache()
    cache.sync(board)
    targets = [Target((1, 1), 0.3, bomb=True), Target((2, 1), 0.3, bomb=True)]

    values = action_values(obs, [BOMB, RIGHT, WAIT], targets, cache, Params())

    assert values[BOMB] > values[RIGHT] >= values[WAIT]


def test_the_agent_bombs_a_crate_it_can_escape_from() -> None:
    state = game_state(arena(crates=[(1, 2)]), (1, 1))
    assert callbacks.act(make_agent(), state) == BOMB


def test_the_agent_never_bombs_for_nothing() -> None:
    agent = make_agent()
    state = game_state(arena(), (7, 7))
    assert all(callbacks.act(agent, state) != BOMB for _ in range(100))
