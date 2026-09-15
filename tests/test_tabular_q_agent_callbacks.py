"""Tests for ``tabular_q_agent``'s callbacks at plan step Q0 (safe-random)."""

import inspect
import json
import logging
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, cast

import pytest

import agents
from agent_code.tabular_q_agent import callbacks, train
from agent_code.tabular_q_agent.config import ENV_VAR
from tests.bfs_boards import arena, game_state, parse_board
from tournament.engine import (
    WorldConfig,
    quiet_logging,
    reset_framework_logging,
    run_round,
)

# Bombing at the closed end: the blast covers the whole corridor.
DEAD_END = parse_board(["######", "#....#", "######"])
# Bombing at (1, 1): RIGHT, RIGHT, DOWN leaves the blast in time.
CORNER = parse_board(["#####", "#...#", "###.#", "#####"])
CORRIDOR = parse_board(["#####", "#...#", "#####"])


@pytest.fixture(autouse=True)
def _defaults(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv(ENV_VAR, raising=False)
    reset_framework_logging()
    yield
    reset_framework_logging()


def make_self(monkeypatch: pytest.MonkeyPatch, **params: object) -> callbacks.AgentSelf:
    if params:
        monkeypatch.setenv(ENV_VAR, json.dumps(params))
    namespace = SimpleNamespace(logger=logging.getLogger("tabular_q_test"), train=False)
    agent = cast(callbacks.AgentSelf, namespace)
    callbacks.setup(agent)
    return agent


def draws(agent: callbacks.AgentSelf, state: dict[str, Any], n: int = 100) -> set[str]:
    return {callbacks.act(agent, cast(callbacks.GameState, state)) for _ in range(n)}


@pytest.mark.parametrize(
    ("module", "api"), [(callbacks, "callbacks"), (train, "train")]
)
def test_callbacks_match_the_framework_api(module: object, api: str) -> None:
    expected: dict[str, list[str]] = agents.AGENT_API[api]
    for name, params in expected.items():
        function = getattr(module, name)
        assert len(inspect.signature(function).parameters) == len(params)


def test_bomb_is_never_played_in_a_dead_end(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = make_self(monkeypatch, seed=0)
    assert draws(agent, game_state(DEAD_END, (1, 1))) == {"RIGHT", "WAIT"}


def test_bomb_is_played_where_it_can_be_escaped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = make_self(monkeypatch, seed=0)
    assert draws(agent, game_state(CORNER, (1, 1))) == {"RIGHT", "WAIT", "BOMB"}


def test_legal_mask_allows_the_fatal_bomb(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = make_self(monkeypatch, seed=0, mask="legal")
    assert draws(agent, game_state(DEAD_END, (1, 1))) == {"RIGHT", "WAIT", "BOMB"}


def test_never_moves_into_a_live_explosion(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = make_self(monkeypatch, seed=0)
    state = game_state(
        CORRIDOR, (2, 1), can_bomb=False, explosions=[((1, 1), 1), ((3, 1), 1)]
    )
    assert draws(agent, state) == {"WAIT"}


def test_the_seed_fixes_the_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    state = cast(callbacks.GameState, game_state(arena(), (7, 7)))
    first = make_self(monkeypatch, seed=3)
    second = make_self(monkeypatch, seed=3)
    assert [callbacks.act(first, state) for _ in range(30)] == [
        callbacks.act(second, state) for _ in range(30)
    ]


def test_plays_a_real_round() -> None:
    config = WorldConfig(
        lineup=("tabular_q_agent", "rule_based_agent"), scenario="classic", seed=0
    )
    with quiet_logging():
        result = run_round(config)
    assert result.steps >= 1
    assert result.agents[0].code_name == "tabular_q_agent"
