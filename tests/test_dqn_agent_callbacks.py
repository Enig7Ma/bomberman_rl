"""D0 safe-random behavior on diagnostic boards and the real engine."""

import inspect
import json
import logging
import random
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, cast

import pytest

import agents
from agent_code.dqn_agent import callbacks
from agent_code.dqn_agent.config import ENV_VAR
from tests.bfs_boards import arena, game_state, parse_board
from tournament.engine import (
    WorldConfig,
    quiet_logging,
    reset_framework_logging,
    run_round,
)

DEAD_END = parse_board(["######", "#....#", "######"])
CORNER = parse_board(["#####", "#...#", "###.#", "#####"])
CORRIDOR = parse_board(["#####", "#...#", "#####"])


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv(ENV_VAR, raising=False)
    reset_framework_logging()
    yield
    reset_framework_logging()


def make_self(monkeypatch: pytest.MonkeyPatch, **params: object) -> callbacks.AgentSelf:
    monkeypatch.setenv(ENV_VAR, json.dumps({"seed": 0, **params}))
    agent = cast(
        callbacks.AgentSelf,
        SimpleNamespace(
            logger=logging.getLogger("dqn_test"),
            train=False,
        ),
    )
    callbacks.setup(agent)
    return agent


def draws(agent: callbacks.AgentSelf, state: dict[str, Any]) -> set[str]:
    return {callbacks.act(agent, state) for _ in range(100)}


def test_framework_callback_signatures() -> None:
    expected: dict[str, list[str]] = agents.AGENT_API["callbacks"]
    for name, params in expected.items():
        assert len(inspect.signature(getattr(callbacks, name)).parameters) == len(
            params
        )


@pytest.mark.parametrize("policy", ["random", "learned"])
def test_only_safe_actions_are_selected(
    monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    agent = make_self(monkeypatch, policy=policy)
    assert draws(agent, game_state(DEAD_END, (1, 1))) == {"RIGHT", "WAIT"}
    assert draws(agent, game_state(CORNER, (1, 1))) == {"RIGHT", "WAIT", "BOMB"}
    state = game_state(
        CORRIDOR, (2, 1), can_bomb=False, explosions=[((1, 1), 1), ((3, 1), 1)]
    )
    assert draws(agent, state) == {"WAIT"}


def test_legal_mask_can_allow_a_fatal_bomb(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = make_self(monkeypatch, mask="legal")
    assert draws(agent, game_state(DEAD_END, (1, 1))) == {"RIGHT", "WAIT", "BOMB"}


def test_seed_is_private_and_repeatable(monkeypatch: pytest.MonkeyPatch) -> None:
    before = random.getstate()
    first = make_self(monkeypatch, seed=3)
    second = make_self(monkeypatch, seed=3)
    state = game_state(arena(), (7, 7))
    assert [callbacks.act(first, state) for _ in range(40)] == [
        callbacks.act(second, state) for _ in range(40)
    ]
    assert random.getstate() == before


def test_a_new_round_resets_extraction_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = make_self(monkeypatch)
    state = game_state(arena(), (7, 7))
    callbacks.act(agent, state)
    previous = agent.extractor
    state["round"] = 2
    callbacks.act(agent, state)
    assert agent.round == 2
    assert agent.extractor is not previous


def test_learned_setting_reports_the_scaffold(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    make_self(monkeypatch, policy="learned")
    assert "no Q-network" in caplog.text


def test_plays_a_real_round() -> None:
    with quiet_logging():
        result = run_round(
            WorldConfig(
                lineup=("dqn_agent", "rule_based_agent"),
                scenario="classic",
                seed=0,
            )
        )
    assert result.steps >= 1
    assert result.agents[0].code_name == "dqn_agent"
    assert result.agents[0].timeouts == 0
