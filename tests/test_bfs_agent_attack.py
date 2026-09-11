"""Tests for ``bfs_agent``'s opponent-pressure evaluation."""

import logging
from collections.abc import Sequence
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import numpy as np
from numpy.typing import NDArray

from agent_code.bfs_agent import callbacks
from agent_code.bfs_agent.attack import (
    bomb_attack_value,
    hunt_targets,
    opponent_escapes,
)
from agent_code.bfs_agent.params import Params
from agent_code.bfs_agent.planning import DistanceCache
from agent_code.bfs_agent.safety import Board
from agent_code.bfs_agent.world_model import (
    BOMB,
    DOWN,
    Geometry,
    Observation,
    Pos,
    Timeline,
    danger_timeline,
    own_bomb,
)
from tests.bfs_boards import arena, game_state, parse_board

# A corridor x = 1..5 with a side exit south at x = 3.
POCKET = parse_board(
    [
        "#######",
        "#.....#",
        "###.###",
        "#######",
    ]
)
PARAMS = Params(trap_value=4.0, pressure_value=0.3)


def situation(
    field: NDArray[np.int64],
    me: Pos,
    others: Sequence[Pos],
    *,
    bombs: Sequence[tuple[Pos, int]] = (),
    others_can_bomb: bool = True,
) -> tuple[Observation, Board, Timeline]:
    obs = Observation.from_game_state(
        game_state(
            field, me, others=others, bombs=bombs, others_can_bomb=others_can_bomb
        )
    )
    geometry = Geometry(obs.field)
    board = Board(obs, geometry)
    timeline = danger_timeline(geometry, board.crates, obs.bombs, obs.explosion_map)
    return obs, board, timeline


def test_a_bomb_at_the_mouth_of_a_dead_end_traps_the_opponent_inside() -> None:
    obs, board, timeline = situation(POCKET, (2, 1), [(1, 1)])

    armed = timeline.with_bombs(board.geometry, board.crates, [own_bomb((2, 1))])
    assert not opponent_escapes(board, armed, (1, 1), frozenset({(2, 1)}))
    assert bomb_attack_value(obs, board, timeline, PARAMS) == PARAMS.trap_value


def test_an_opponent_that_can_run_is_only_under_pressure() -> None:
    # From (4, 1) the opponent reaches the side exit (3, 2) in two moves.
    obs, board, timeline = situation(POCKET, (2, 1), [(4, 1)])
    assert bomb_attack_value(obs, board, timeline, PARAMS) == PARAMS.pressure_value


def test_no_credit_for_an_opponent_that_is_doomed_anyway() -> None:
    # Someone else's bomb, about to explode, already covers the whole pocket.
    obs, board, timeline = situation(POCKET, (2, 1), [(1, 1)], bombs=[((1, 1), 0)])
    assert bomb_attack_value(obs, board, timeline, PARAMS) == 0.0


def test_far_opponents_are_worth_nothing() -> None:
    obs, board, timeline = situation(arena(), (1, 1), [(15, 15)])
    assert bomb_attack_value(obs, board, timeline, PARAMS) == 0.0


def test_hunt_targets_are_the_reachable_cells_of_the_opponents_blast() -> None:
    obs, board, _ = situation(arena(), (1, 1), [(1, 5)])
    cache = DistanceCache()
    cache.sync(board)

    cells = {t.pos for t in hunt_targets(obs, board, cache, Params(hunt_radius=5))}

    # The blast of (1, 5) also covers (1, 7), (1, 8), (3, 5) and (4, 5), but
    # those are 6 or 7 steps from us, beyond the radius.
    assert cells == {(1, 2), (1, 3), (1, 4), (1, 6), (2, 5)}
    # Movement targets only; a bomb there is valued by bomb_attack_value.
    assert not any(t.bomb for t in hunt_targets(obs, board, cache, Params()))


def make_agent(**overrides: float) -> callbacks.AgentSelf:
    agent = cast(
        callbacks.AgentSelf,
        SimpleNamespace(logger=logging.getLogger("bfs_agent_test"), train=False),
    )
    callbacks.setup(agent)
    agent.params = replace(agent.params, **overrides)
    return agent


def test_the_agent_bombs_to_trap_an_opponent() -> None:
    # No crates and no coins: the trap is the only reason to bomb here.
    state = game_state(POCKET, (2, 1), others=[(1, 1)], others_can_bomb=False)
    assert callbacks.act(make_agent(), state) == BOMB


def test_the_agent_does_not_bomb_without_attack_value() -> None:
    state = game_state(POCKET, (2, 1), others=[(1, 1)], others_can_bomb=False)
    agent = make_agent(trap_value=0.0, pressure_value=0.0)
    assert all(callbacks.act(agent, state) != BOMB for _ in range(50))


def test_with_nothing_else_to_do_the_agent_hunts() -> None:
    # The opponent's blast reaches (1, 6) and (1, 7), 5 and 6 steps south.
    state = game_state(arena(), (1, 1), others=[(1, 9)], others_can_bomb=False)
    agent = make_agent()
    assert all(callbacks.act(agent, state) == DOWN for _ in range(20))
