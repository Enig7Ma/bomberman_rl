"""Frozen schedule coverage and private opponent-stream pairing."""

import importlib
import inspect
import json
import logging
import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from docs.experiments.d12_heldout import PROTOCOL, opponent_streams, schedule


def test_frozen_schedule_has_unique_games_and_separate_latency() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    games = schedule(protocol, "heldout")
    assert len(games) == len({g["id"] for g in games}) == 17400
    assert {g["seed"] for g in games} == set(range(3000, 3100))
    h2h = [g for g in games if g["candidate"] == "head_to_head"]
    assert len(h2h) == 1200
    assert all(len(g["identities"]) == 2 for g in h2h)
    for name in [c["id"] for c in protocol["candidates"]] + ["reference"]:
        assert sum(g["candidate"] == name for g in games) == 1800
    assert len(schedule(protocol, "latency")) == 50
    assert {g["seed"] for g in schedule(protocol, "smoke")} == {500}


def test_opponent_streams_pair_actual_seats_and_restore_globals() -> None:
    module = importlib.import_module("agent_code.rule_based_agent.callbacks")

    def setup(agent: object) -> None:
        np.random.seed()
        random.random()

    def act(agent: object, state: dict[str, object]) -> str:
        return f"{np.random.random()}:{random.random()}"

    before_np, before_py = np.random.get_state(), random.getstate()

    def sample(lineup: tuple[str, ...]) -> dict[int, list[str]]:
        with opponent_streams(lineup, 500, {}):
            assert len(inspect.signature(module.setup).parameters) == 1
            assert len(inspect.signature(module.act).parameters) == 2
            result: dict[int, list[str]] = {}
            for seat, name in enumerate(lineup):
                if name == "rule_based_agent":
                    a = SimpleNamespace()
                    module.setup(a)
                    result[seat] = [module.act(a, {}) for _ in range(10)]
            return result

    with patch.object(module, "setup", setup), patch.object(module, "act", act):
        reference = sample(("rule_based_agent",) * 4)
        candidate = sample(
            ("rule_based_agent", "dqn_agent", "rule_based_agent", "rule_based_agent")
        )
        assert candidate == {k: v for k, v in reference.items() if k != 1}
        assert candidate[0] != candidate[2]
    after = np.random.get_state()
    np.testing.assert_array_equal(before_np[1], after[1])
    assert before_np[0] == after[0] and before_np[2:] == after[2:]
    assert before_py == random.getstate()


def test_mixed_lineup_assigns_seats_before_callbacks_execute() -> None:
    lineup = ("rule_based_agent", "coin_collector_agent", "peaceful_agent")
    with opponent_streams(lineup, 500, {}):
        for name in lineup:
            module = importlib.import_module(f"agent_code.{name}.callbacks")
            module.setup(SimpleNamespace(logger=logging.getLogger("test")))
