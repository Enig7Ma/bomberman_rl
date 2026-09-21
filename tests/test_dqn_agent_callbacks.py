"""Safe-random fallback behavior on diagnostic boards and the real engine."""

import inspect
import json
import logging
import random
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

import agents
from agent_code.dqn_agent import callbacks, config
from agent_code.dqn_agent.config import ENV_VAR, MODEL_ENV_VAR, Config
from agent_code.dqn_agent.core.world_model import ACTIONS, Observation
from agent_code.dqn_agent.encoder import ENCODERS
from agent_code.dqn_agent.network import QNetwork
from agent_code.dqn_agent.symmetry import IDENTITY, canonical, to_canonical
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
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
    monkeypatch.setattr(config, "DEFAULT_MODEL_PATH", tmp_path / "default.npz")
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


def test_missing_default_reports_safe_random_fallback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    make_self(monkeypatch, policy="learned")
    assert "no Q-network" in caplog.text
    assert "playing safe-random" in caplog.text


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


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("fault", ["missing", "corrupt", "schema"])
def test_loading_rules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    explicit: bool,
    fault: str,
) -> None:
    path = tmp_path / "default.npz"
    if explicit:
        monkeypatch.setenv(MODEL_ENV_VAR, str(path))
    if fault == "corrupt":
        path.write_bytes(b"broken archive")
    elif fault == "schema":
        net = QNetwork.random(ENCODERS[Config().encoder], 0)
        net.header["schema_id"] = "old schema"
        net.save(path)
    if explicit:
        with pytest.raises(FileNotFoundError if fault == "missing" else ValueError):
            make_self(monkeypatch)
    else:
        agent = make_self(monkeypatch)
        assert agent.q_function is None
        assert "playing safe-random" in caplog.text
        assert callbacks.act(agent, game_state(CORNER, (1, 1))) in {
            "RIGHT",
            "WAIT",
            "BOMB",
        }


@pytest.mark.parametrize("explicit", [False, True])
def test_missing_training_model_uses_init_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    explicit: bool,
) -> None:
    if explicit:
        monkeypatch.setenv(MODEL_ENV_VAR, str(tmp_path / "new.npz"))
    monkeypatch.setenv(ENV_VAR, '{"init_seed": 14}')
    agent = cast(
        callbacks.AgentSelf,
        SimpleNamespace(logger=logging.getLogger("test"), train=True),
    )
    callbacks.setup(agent)
    assert isinstance(agent.q_function, QNetwork)
    expected = QNetwork.random(ENCODERS[Config().encoder], 14)
    for name, array in expected.arrays.items():
        np.testing.assert_array_equal(agent.q_function.arrays[name], array)
    assert not agent.model_file.exists()
    agent.model_file.write_bytes(b"broken")
    with pytest.raises(ValueError):
        callbacks.setup(agent)


@pytest.mark.parametrize("wanted", ["WAIT", "BOMB", "RIGHT"])
@pytest.mark.parametrize("explicit", [False, True])
def test_loaded_network_uses_canonical_mask_and_maps_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wanted: str,
    explicit: bool,
) -> None:
    # Real corner: RIGHT/WAIT/BOMB allowed, canonical rotation changes RIGHT.
    state = game_state(CORNER, (1, 1))
    probe = make_self(monkeypatch, policy="random")
    extracted = probe.extractor.extract(Observation.from_game_state(state))
    index, symmetry = canonical(extracted.features, probe.encoding)
    assert symmetry != IDENTITY
    assert to_canonical("RIGHT", symmetry) != "RIGHT"
    net = QNetwork.random(ENCODERS[Config().encoder], 0)
    for array in net.arrays.values():
        array.fill(0)
    net.arrays["b2"][ACTIONS.index(to_canonical(wanted, symmetry))] = 5
    forbidden = next(a for a in ACTIONS if a not in extracted.allowed)
    net.arrays["b2"][ACTIONS.index(to_canonical(forbidden, symmetry))] = 100
    path = tmp_path / "default.npz"
    net.save(path)
    if explicit:
        monkeypatch.setenv(MODEL_ENV_VAR, str(path))
    agent = make_self(monkeypatch)
    assert isinstance(agent.q_function, QNetwork)
    assert callbacks.act(agent, state) == wanted

    class Probe:
        def values(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
            np.testing.assert_array_equal(
                x,
                agent.encoder.encode(agent.encoding.decode(index), extracted, symmetry),
            )
            return net.values(x)

    agent.q_function = Probe()
    assert callbacks.act(agent, state) == wanted
    random_agent = make_self(monkeypatch, policy="random")
    assert draws(random_agent, state) == {"RIGHT", "WAIT", "BOMB"}


def test_loaded_ties_use_private_agent_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    net = QNetwork.random(ENCODERS[Config().encoder], 0)
    for array in net.arrays.values():
        array.fill(0)
    net.save(tmp_path / "default.npz")
    state = game_state(CORNER, (1, 1))
    before = random.getstate()
    actions = {
        callbacks.act(make_self(monkeypatch, seed=seed), state) for seed in range(20)
    }
    assert actions == {"RIGHT", "WAIT", "BOMB"}
    first, second = make_self(monkeypatch, seed=3), make_self(monkeypatch, seed=3)
    assert [callbacks.act(first, state) for _ in range(20)] == [
        callbacks.act(second, state) for _ in range(20)
    ]
    assert random.getstate() == before
