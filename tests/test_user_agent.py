"""Tests for the ``user_agent`` skeleton: its API shape and its keyboard policy."""

import inspect
import logging
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

import agents
from agent_code.user_agent import callbacks, train
from agent_code.user_agent.callbacks import ACTIONS, AgentSelf, GameState
from environment import BombeRLeWorld, WorldArgs
from tournament.engine import ensure_repo_cwd, quiet_logging, reset_framework_logging


def make_self(*, training: bool = False) -> AgentSelf:
    agent = SimpleNamespace(logger=logging.getLogger("user_agent_test"), train=training)
    typed = cast(AgentSelf, agent)
    callbacks.setup(typed)
    return typed


def make_state(user_input: str | None) -> GameState:
    return {
        "round": 1,
        "step": 1,
        "field": np.zeros((17, 17), dtype=np.int64),
        "self": ("me", 0, True, (1, 1)),
        "others": [],
        "bombs": [],
        "coins": [],
        "explosion_map": np.zeros((17, 17)),
        "user_input": user_input,
    }


@pytest.mark.parametrize(
    ("module", "api"),
    [(callbacks, "callbacks"), (train, "train")],
)
def test_callbacks_match_the_framework_api(module: object, api: str) -> None:
    expected: dict[str, list[str]] = agents.AGENT_API[api]
    for name, params in expected.items():
        function = getattr(module, name)
        assert len(inspect.signature(function).parameters) == len(params)


@pytest.mark.parametrize("key", ACTIONS)
def test_act_plays_the_pressed_key(key: str) -> None:
    assert callbacks.act(make_self(), make_state(key)) == key


@pytest.mark.parametrize("key", [None, "", "JUMP"])
def test_act_waits_without_a_valid_key(key: str | None) -> None:
    assert callbacks.act(make_self(), make_state(key)) == "WAIT"


def test_a_training_round_runs_through_the_engine() -> None:
    """Loads both modules via the real framework and plays one round."""
    ensure_repo_cwd()
    args = WorldArgs(
        no_gui=True,
        fps=15,
        turn_based=False,
        update_interval=0.1,
        save_replay=False,
        replay=None,
        make_video=False,
        continue_without_training=True,
        log_dir="logs",
        save_stats=False,
        match_name=None,
        seed=0,
        silence_errors=False,
        scenario="coin-heaven",
    )
    try:
        with quiet_logging():
            world = BombeRLeWorld(
                args, [("user_agent", True), ("peaceful_agent", False)]
            )
            world.new_round()
            while world.running:
                world.do_step()
        assert world.step >= 1
    finally:
        reset_framework_logging()
