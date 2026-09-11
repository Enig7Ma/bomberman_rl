"""Tests for ``bfs_agent``'s observation parsing, legal actions and callbacks."""

import logging
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from agent_code.bfs_agent import callbacks
from agent_code.bfs_agent.world_model import (
    ACTIONS,
    BOMB,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    WAIT,
    Observation,
    legal_actions,
)
from tests.bfs_boards import arena, game_state, parse_board

CORRIDOR = parse_board(
    [
        "#####",
        "#...#",
        "#####",
    ]
)


def make_self() -> callbacks.AgentSelf:
    agent = SimpleNamespace(logger=logging.getLogger("bfs_agent_test"), train=False)
    typed = cast(callbacks.AgentSelf, agent)
    callbacks.setup(typed)
    return typed


def test_observation_converts_numpy_coordinates_to_int() -> None:
    state = game_state(
        arena(),
        (np.int64(1), np.int64(1)),  # pyright: ignore[reportArgumentType]
        coins=[(np.int64(3), np.int64(1))],  # pyright: ignore[reportArgumentType]
    )
    obs = Observation.from_game_state(state)

    assert obs.me.pos == (1, 1)
    assert type(obs.me.pos[0]) is int
    assert type(obs.coins[0][0]) is int


def test_open_tile_allows_every_move() -> None:
    obs = Observation.from_game_state(game_state(arena(), (3, 3)))
    assert set(legal_actions(obs)) == {UP, RIGHT, DOWN, LEFT, WAIT, BOMB}


def test_board_edge_corner_only_allows_inward_moves() -> None:
    obs = Observation.from_game_state(game_state(arena(), (1, 1)))
    assert set(legal_actions(obs)) == {RIGHT, DOWN, WAIT, BOMB}


def test_crates_bombs_and_agents_block_moves() -> None:
    field = arena(crates=[(4, 3)])
    obs = Observation.from_game_state(
        game_state(field, (3, 3), others=[(3, 2)], bombs=[((2, 3), 2)])
    )
    # right: crate, up: agent, left: bomb
    assert set(legal_actions(obs)) == {DOWN, WAIT, BOMB}


def test_bomb_needs_an_available_bomb() -> None:
    obs = Observation.from_game_state(game_state(CORRIDOR, (2, 1), can_bomb=False))
    assert set(legal_actions(obs)) == {LEFT, RIGHT, WAIT}


def test_act_never_moves_into_a_live_explosion() -> None:
    agent = make_self()
    state = game_state(CORRIDOR, (2, 1), explosions=[((1, 1), 1), ((3, 1), 1)])
    for _ in range(50):
        assert callbacks.act(agent, state) == WAIT


@pytest.mark.parametrize("me", [(1, 1), (3, 3), (15, 15), (7, 1)])
def test_act_returns_a_legal_action(me: tuple[int, int]) -> None:
    agent = make_self()
    state = game_state(arena(crates=[(2, 1), (1, 2)]), me)
    legal = set(legal_actions(Observation.from_game_state(state)))
    for _ in range(20):
        action = callbacks.act(agent, state)
        assert action in ACTIONS
        assert action in legal


def test_a_new_round_resets_the_route() -> None:
    """The stock framework reuses one agent for every round."""
    agent = make_self()
    callbacks.act(agent, game_state(arena(), (1, 1), coins=[(4, 1)], round_number=1))
    first = agent.route
    assert first.target == (4, 1)

    callbacks.act(agent, game_state(arena(), (1, 1), coins=[(4, 1)], round_number=1))
    assert agent.route is first

    callbacks.act(agent, game_state(arena(), (1, 1), coins=[(1, 4)], round_number=2))
    assert agent.route is not first
    assert agent.route.target == (1, 4)
