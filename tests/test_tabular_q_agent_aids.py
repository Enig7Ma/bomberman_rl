"""Tests for what Q7-Q10 added to ``tabular_q_agent``.

- ``bomb_aid``: paid once per confirmed bomb, per live crate it will destroy;
- ``policy="heuristic"``: the fixed-priority control plays from the agent;
- ``symmetry=false``: raw feature indices instead of canonical rows.
"""

import json
import logging
import random
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent_code.tabular_q_agent import callbacks, config, train
from agent_code.tabular_q_agent.config import ENV_VAR, MODEL_ENV_VAR, Config
from agent_code.tabular_q_agent.core.world_model import Observation
from agent_code.tabular_q_agent.features import ENCODINGS, Extractor
from agent_code.tabular_q_agent.learner import Learner, Selection
from agent_code.tabular_q_agent.qtable import ACTION_COLUMN, QTable
from agent_code.tabular_q_agent.rewards import Rewards
from agent_code.tabular_q_agent.symmetry import IDENTITY, canonical
from agent_code.tabular_q_agent.transitions import Trainer
from tests.bfs_boards import arena, game_state, parse_board

UP, RIGHT, DOWN, LEFT, WAIT, BOMB = range(6)
GAMMA = 0.9
CORNER = parse_board(["#####", "#...#", "###.#", "#####"])


# --- bomb_aid ----------------------------------------------------------------


def make_trainer(bomb_aid: float = 0.1) -> Trainer:
    learner = Learner(
        QTable.zeros(ENCODINGS["E1"]),
        gamma=GAMMA,
        alpha_omega=0.7,
        alpha_min=0.05,
        rng=random.Random(0),
    )
    trainer = Trainer(learner, Rewards(gamma=GAMMA, bomb_aid=bomb_aid), epsilon=0.0)
    trainer.begin_round(1, [])
    return trainer


def act(
    trainer: Trainer, step: int, state: int, action: int, played: str, hits: int
) -> None:
    trainer.observe(1, step, state, (UP, WAIT, BOMB), None, bomb_hits=hits)
    trainer.chose(Selection(action, explored=False, unseen=False), played)


def test_a_confirmed_bomb_earns_the_aid_per_crate() -> None:
    trainer = make_trainer()
    act(trainer, 1, 0, BOMB, "BOMB", hits=3)
    trainer.events_occurred(1, 1, "BOMB", ["BOMB_DROPPED"])
    act(trainer, 2, 1, UP, "UP", hits=0)  # completes the bomb transition
    assert trainer.learner.table.q[0, BOMB] == pytest.approx(0.3)


def test_a_bomb_that_was_not_dropped_earns_nothing() -> None:
    trainer = make_trainer()
    act(trainer, 1, 0, BOMB, "BOMB", hits=3)
    trainer.events_occurred(1, 1, "BOMB", ["INVALID_ACTION"])
    act(trainer, 2, 1, UP, "UP", hits=0)
    assert trainer.learner.table.q[0, BOMB] == 0.0


def test_a_bomb_on_the_final_step_is_paid_once() -> None:
    trainer = make_trainer()
    act(trainer, 1, 0, BOMB, "BOMB", hits=2)
    trainer.events_occurred(1, 1, "BOMB", ["BOMB_DROPPED"])
    trainer.finish("BOMB", ["BOMB_DROPPED", "SURVIVED_ROUND"])
    assert trainer.learner.table.q[0, BOMB] == pytest.approx(0.2)


def test_a_bomb_on_the_death_step_is_paid_from_end_of_round() -> None:
    trainer = make_trainer()
    act(trainer, 1, 0, BOMB, "BOMB", hits=4)
    trainer.finish("BOMB", ["BOMB_DROPPED", "GOT_KILLED"])
    assert trainer.learner.table.q[0, BOMB] == pytest.approx(0.4)


def test_the_bomb_aid_is_off_by_default() -> None:
    assert Rewards(gamma=GAMMA).bomb(5) == 0.0
    assert Rewards(gamma=GAMMA, bomb_aid=0.25).bomb(4) == 1.0


def test_crate_distance_is_the_walk_to_the_chosen_spot() -> None:
    extractor = Extractor(ENCODINGS["E3"], "best_tier", random.Random(0))
    field = arena(crates=[(1, 2)])
    # From (3, 1) the best spot is (1, 1), two steps west; on it, zero.
    away = extractor.extract(Observation.from_game_state(game_state(field, (3, 1))))
    here = extractor.extract(Observation.from_game_state(game_state(field, (1, 1))))
    none = extractor.extract(Observation.from_game_state(game_state(arena(), (1, 1))))
    assert (away.crate_distance, here.crate_distance, none.crate_distance) == (
        2,
        0,
        None,
    )


def test_the_spot_potential_adds_to_the_coin_potential() -> None:
    rewards = Rewards(gamma=GAMMA, coin_potential=0.5, spot_potential=0.2)
    assert rewards.potential(None, None) == 0.0
    assert rewards.potential(1, None) == pytest.approx(0.25)
    assert rewards.potential(None, 3) == pytest.approx(0.05)
    assert rewards.potential(1, 0) == pytest.approx(0.45)
    assert Rewards(gamma=GAMMA).potential(None, 0) == 0.0  # off by default


def test_a_spot_potential_reaches_the_update_through_observe() -> None:
    learner = Learner(
        QTable.zeros(ENCODINGS["E1"]),
        gamma=GAMMA,
        alpha_omega=0.7,
        alpha_min=0.05,
        rng=random.Random(0),
    )
    rewards = Rewards(gamma=GAMMA, spot_potential=0.2)
    trainer = Trainer(learner, rewards, epsilon=0.0)
    trainer.begin_round(1, [])
    trainer.observe(1, 1, 0, (UP, WAIT), None, crate_distance=3)  # phi 0.05
    trainer.chose(Selection(UP, explored=False, unseen=False), "UP")
    trainer.events_occurred(1, 1, "UP", ["MOVED_UP"])
    trainer.observe(1, 2, 1, (UP, WAIT), None, crate_distance=1)  # phi' 0.1
    assert learner.table.q[0, UP] == pytest.approx(GAMMA * 0.1 - 0.05)


def test_bomb_hits_is_the_uncapped_yield() -> None:
    field = arena(crates=[(2, 1), (3, 1), (4, 1), (1, 2), (1, 3)])
    obs = Observation.from_game_state(game_state(field, (1, 1)))
    extracted = Extractor(ENCODINGS["E3"], "best_tier", random.Random(0)).extract(obs)
    assert (extracted.bomb_hits, extracted.features.bomb_yield) == (5, 3)


# --- config ------------------------------------------------------------------------


def test_new_fields_default_and_validate() -> None:
    assert (Config().bomb_aid, Config().symmetry) == (0.0, True)
    parsed = Config.from_env(
        {ENV_VAR: '{"bomb_aid": 0.1, "symmetry": false, "policy": "heuristic"}'}
    )
    assert (parsed.bomb_aid, parsed.symmetry, parsed.policy) == (
        0.1,
        False,
        "heuristic",
    )
    assert Config.from_env({ENV_VAR: '{"spot_potential": 0.2}'}).spot_potential == 0.2
    for bad in (
        '{"bomb_aid": "x"}',
        '{"symmetry": 0}',
        '{"policy": "greedy"}',
        '{"spot_potential": -0.1}',
    ):
        with pytest.raises(ValueError):
            Config.from_env({ENV_VAR: bad})


# --- policies in the agent -----------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
    monkeypatch.setattr(config, "DEFAULT_MODEL_PATH", tmp_path / "shipped.npz")
    yield


def make_self(
    monkeypatch: pytest.MonkeyPatch, model: Path | None = None, **params: object
) -> callbacks.AgentSelf:
    monkeypatch.setenv(ENV_VAR, json.dumps({"seed": 0, **params}))
    if model is not None:
        monkeypatch.setenv(MODEL_ENV_VAR, str(model))
    namespace = SimpleNamespace(logger=logging.getLogger("aids_test"), train=False)
    agent = cast(callbacks.AgentSelf, namespace)
    callbacks.setup(agent)
    return agent


def draws(agent: callbacks.AgentSelf, state: dict[str, Any], n: int = 40) -> set[str]:
    return {callbacks.act(agent, cast(callbacks.GameState, state)) for _ in range(n)}


def test_the_heuristic_policy_bombs_a_crate_it_can_escape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = make_self(monkeypatch, policy="heuristic")
    assert draws(agent, game_state(arena(crates=[(1, 2)]), (1, 1))) == {"BOMB"}


def test_without_symmetry_the_table_is_indexed_by_raw_features(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = game_state(CORNER, (1, 1))  # no random tie-breaking in its features
    encoding = ENCODINGS["E3"]
    extractor = Extractor(encoding, "best_tier", random.Random(0))
    features = extractor.extract(Observation.from_game_state(state)).features
    assert canonical(features, encoding)[1] != IDENTITY  # raw and canonical rows differ

    table = QTable.zeros(encoding)
    table.q[encoding.encode(features), ACTION_COLUMN["RIGHT"]] = 1.0
    table.save(tmp_path / "raw.npz")

    raw = make_self(monkeypatch, model=tmp_path / "raw.npz", symmetry=False)
    assert draws(raw, state) == {"RIGHT"}
    shared = make_self(monkeypatch, model=tmp_path / "raw.npz")
    assert draws(shared, state) != {"RIGHT"}  # that row means something else there


# --- teacher-guided exploration (Q11) -------------------------------------------------


def make_training_self(
    monkeypatch: pytest.MonkeyPatch, model: Path, **params: object
) -> callbacks.AgentSelf:
    """An agent in training mode, with its training callbacks set up."""
    monkeypatch.setenv(ENV_VAR, json.dumps({"seed": 0, **params}))
    monkeypatch.setenv(MODEL_ENV_VAR, str(model))
    namespace = SimpleNamespace(logger=logging.getLogger("aids_test"), train=True)
    agent = cast(callbacks.AgentSelf, namespace)
    callbacks.setup(agent)
    train.setup_training(agent)
    return agent


def table_that_always_waits(tmp_path: Path) -> Path:
    table = QTable.zeros(ENCODINGS["E3"])
    table.q[:, ACTION_COLUMN["WAIT"]] = 1.0
    path = tmp_path / "waiting.npz"
    table.save(path)
    return path


def test_the_teacher_acts_for_the_table_while_training(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The table says WAIT everywhere; the heuristic bombs the crate next door.
    model = table_that_always_waits(tmp_path)
    state = cast(callbacks.GameState, game_state(arena(crates=[(1, 2)]), (1, 1)))
    # One act per agent: a second one without its events would be a bookkeeping error.
    for seed in range(3):
        agent = make_training_self(monkeypatch, model, teacher_share=1.0, seed=seed)
        assert callbacks.act(agent, state) == "BOMB"
        assert agent.trainer is not None
        assert agent.trainer.pending is not None
        assert agent.trainer.pending.played == "BOMB"


def test_without_a_teacher_share_training_plays_the_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = table_that_always_waits(tmp_path)
    state = cast(callbacks.GameState, game_state(arena(crates=[(1, 2)]), (1, 1)))
    agent = make_training_self(monkeypatch, model, epsilon=0.0)
    assert callbacks.act(agent, state) == "WAIT"


def test_the_teacher_never_acts_outside_training(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = table_that_always_waits(tmp_path)
    state = game_state(arena(crates=[(1, 2)]), (1, 1))
    agent = make_self(monkeypatch, model=model, teacher_share=1.0)
    assert draws(agent, state) == {"WAIT"}


def test_the_teacher_share_defaults_to_off_and_is_validated() -> None:
    assert Config().teacher_share == 0.0
    assert Config.from_env({ENV_VAR: '{"teacher_share": 0.3}'}).teacher_share == 0.3
    for bad in ('{"teacher_share": 1.5}', '{"teacher_share": -0.1}'):
        with pytest.raises(ValueError):
            Config.from_env({ENV_VAR: bad})
