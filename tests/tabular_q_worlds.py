"""Helpers for engine-driven tests of ``tabular_q_agent`` in training mode."""

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

from agent_code.tabular_q_agent.callbacks import AgentSelf
from agent_code.tabular_q_agent.config import ENV_VAR, METRICS_ENV_VAR, MODEL_ENV_VAR
from agent_code.tabular_q_agent.learner import Selection
from agent_code.tabular_q_agent.qtable import ACTION_COLUMN
from environment import BombeRLeWorld

# The only actions every board symmetry leaves unchanged, so a scripted
# canonical action is also the real one.
SCRIPTABLE = ("WAIT", "BOMB")


def training_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **params: object
) -> tuple[Path, Path]:
    """Point the agent's table and metrics into ``tmp_path``; returns both."""
    model = tmp_path / "q_table.npz"
    metrics = tmp_path / "metrics.jsonl"
    monkeypatch.setenv(MODEL_ENV_VAR, str(model))
    monkeypatch.setenv(METRICS_ENV_VAR, str(metrics))
    monkeypatch.setenv(ENV_VAR, json.dumps({"seed": 0, **params}))
    return model, metrics


def agent_self(world: BombeRLeWorld, seat: int = 0) -> AgentSelf:
    """The ``self`` namespace of the agent in ``seat`` (sequential backend)."""
    agent: Any = world.agents[seat]
    return cast(AgentSelf, agent.backend.runner.fake_self)


def script(
    monkeypatch: pytest.MonkeyPatch, agent: AgentSelf, actions: Iterable[str]
) -> None:
    """Make the agent play ``actions`` in order, then ``WAIT`` forever."""
    planned = list(actions)
    if any(action not in SCRIPTABLE for action in planned):
        raise ValueError(f"only {SCRIPTABLE} can be scripted, got {planned}")
    moves = iter(planned)

    def select(state: int, allowed: Sequence[int], epsilon: float = 0.0) -> Selection:
        return Selection(
            ACTION_COLUMN[next(moves, "WAIT")], explored=False, unseen=False
        )

    monkeypatch.setattr(agent.learner, "select", select)


@dataclass(frozen=True)
class UpdateCall:
    state: int
    action: int
    reward: float
    next_state: int | None


def spy_updates(monkeypatch: pytest.MonkeyPatch, agent: AgentSelf) -> list[UpdateCall]:
    """Record every Q-update the agent's learner applies."""
    calls: list[UpdateCall] = []
    original = agent.learner.update

    def update(
        state: int,
        action: int,
        reward: float,
        next_state: int | None,
        next_allowed: Sequence[int] = (),
    ) -> float:
        calls.append(UpdateCall(state, action, reward, next_state))
        return original(state, action, reward, next_state, next_allowed)

    monkeypatch.setattr(agent.learner, "update", update)
    return calls
