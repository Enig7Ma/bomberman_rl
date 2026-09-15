"""Tests for ``tabular_q_agent``'s callbacks: masked play and table loading."""

import inspect
import json
import logging
import random
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import agents
from agent_code.tabular_q_agent import callbacks, config, train
from agent_code.tabular_q_agent.config import ENV_VAR, MODEL_ENV_VAR
from agent_code.tabular_q_agent.core.world_model import Observation
from agent_code.tabular_q_agent.features import ENCODINGS, Extractor
from agent_code.tabular_q_agent.qtable import (
    ACTION_COLUMN,
    QTable,
    SchemaMismatchError,
)
from agent_code.tabular_q_agent.symmetry import IDENTITY, canonical, to_canonical
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
LOGGER = "tabular_q_test"


@pytest.fixture(autouse=True)
def _defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
    # Never let a table shipped in the agent directory leak into these tests.
    monkeypatch.setattr(config, "DEFAULT_MODEL_PATH", tmp_path / "shipped.npz")
    reset_framework_logging()
    yield
    reset_framework_logging()


def make_self(
    monkeypatch: pytest.MonkeyPatch,
    *,
    training: bool = False,
    model: Path | None = None,
    **params: object,
) -> callbacks.AgentSelf:
    if params:
        monkeypatch.setenv(ENV_VAR, json.dumps(params))
    if model is not None:
        monkeypatch.setenv(MODEL_ENV_VAR, str(model))
    namespace = SimpleNamespace(logger=logging.getLogger(LOGGER), train=training)
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


# --- the mask, under both policies -----------------------------------------
# With no table the learned policy sees only ties, so it samples the mask
# uniformly just like the random control.

POLICIES = ["learned", "random"]


@pytest.mark.parametrize("policy", POLICIES)
def test_bomb_is_never_played_in_a_dead_end(
    monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    agent = make_self(monkeypatch, seed=0, policy=policy)
    assert draws(agent, game_state(DEAD_END, (1, 1))) == {"RIGHT", "WAIT"}


@pytest.mark.parametrize("policy", POLICIES)
def test_bomb_is_played_where_it_can_be_escaped(
    monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    agent = make_self(monkeypatch, seed=0, policy=policy)
    assert draws(agent, game_state(CORNER, (1, 1))) == {"RIGHT", "WAIT", "BOMB"}


def test_legal_mask_allows_the_fatal_bomb(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = make_self(monkeypatch, seed=0, mask="legal")
    assert draws(agent, game_state(DEAD_END, (1, 1))) == {"RIGHT", "WAIT", "BOMB"}


@pytest.mark.parametrize("policy", POLICIES)
def test_never_moves_into_a_live_explosion(
    monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    agent = make_self(monkeypatch, seed=0, policy=policy)
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


# --- playing from a table --------------------------------------------------


@pytest.mark.parametrize("favourite", ["RIGHT", "WAIT", "BOMB"])
def test_the_learned_policy_plays_the_tables_best_action(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, favourite: str
) -> None:
    # The corner state has no coins, crates or opponents, so its features
    # involve no random tie-breaking and the test can reproduce its row.
    state = game_state(CORNER, (1, 1))
    encoding = ENCODINGS["E3"]
    extractor = Extractor(encoding, "best_tier", random.Random(0))
    features = extractor.extract(Observation.from_game_state(state)).features
    row, symmetry = canonical(features, encoding)
    # Only RIGHT is allowed among the moves, and the canonical row stores it
    # under another direction -- so this also checks the mapping back.
    assert symmetry != IDENTITY

    table = QTable.zeros(encoding)
    table.q[row, ACTION_COLUMN[to_canonical(favourite, symmetry)]] = 1.0
    table.save(tmp_path / "q.npz")

    agent = make_self(monkeypatch, seed=0, model=tmp_path / "q.npz")
    assert draws(agent, state, n=30) == {favourite}


def test_the_random_policy_ignores_the_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = game_state(CORNER, (1, 1))
    table = QTable.zeros(ENCODINGS["E3"])
    table.q[:, ACTION_COLUMN["BOMB"]] = 1.0
    table.save(tmp_path / "q.npz")

    agent = make_self(monkeypatch, seed=0, policy="random", model=tmp_path / "q.npz")
    assert draws(agent, state) == {"RIGHT", "WAIT", "BOMB"}


def test_an_explicit_model_must_exist_outside_training(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(FileNotFoundError):
        make_self(monkeypatch, model=tmp_path / "missing.npz")


def test_training_may_start_from_a_model_that_does_not_exist_yet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent = make_self(monkeypatch, training=True, model=tmp_path / "new.npz")
    assert agent.table.visited_states == 0


def test_an_explicit_model_for_another_encoding_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    QTable.zeros(ENCODINGS["E1"]).save(tmp_path / "q.npz")
    with pytest.raises(SchemaMismatchError):
        make_self(monkeypatch, model=tmp_path / "q.npz")


def test_a_missing_shipped_table_is_logged_not_fatal(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    agent = make_self(monkeypatch)
    assert agent.table.visited_states == 0
    assert any(
        record.levelno == logging.ERROR and "no Q-table" in record.getMessage()
        for record in caplog.records
    )


def test_a_broken_shipped_table_is_logged_not_fatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    broken = tmp_path / "broken.npz"
    broken.write_bytes(b"not a table")
    monkeypatch.setattr(config, "DEFAULT_MODEL_PATH", broken)

    agent = make_self(monkeypatch)
    assert agent.table.visited_states == 0
    assert any(
        record.levelno == logging.ERROR and "cannot use" in record.getMessage()
        for record in caplog.records
    )


def test_plays_a_real_round() -> None:
    config_ = WorldConfig(
        lineup=("tabular_q_agent", "rule_based_agent"), scenario="classic", seed=0
    )
    with quiet_logging():
        result = run_round(config_)
    assert result.steps >= 1
    assert result.agents[0].code_name == "tabular_q_agent"
