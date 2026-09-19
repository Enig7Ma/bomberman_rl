"""Candidate validation fixes opponent streams by seat, not lineup ordinal."""

import importlib
import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from docs.experiments.dqn_candidates import rule_streams


def test_rule_streams_pair_opponents_and_preserve_global_rng() -> None:
    module = importlib.import_module("agent_code.rule_based_agent.callbacks")

    def setup(agent: object) -> None:
        np.random.seed()
        random.random()

    def act(agent: object, state: dict[str, object]) -> str:
        return f"{np.random.random()}:{random.random()}"

    before_np, before_py = np.random.get_state(), random.getstate()

    def sample(reference: bool) -> dict[int, list[str]]:
        with rule_streams(500, reference) as seats:
            agents = [SimpleNamespace() for _ in range(4 if reference else 3)]
            for a in agents:
                module.setup(a)
            return {
                seats[id(a)]: [module.act(a, {}) for _ in range(10)] for a in agents
            }

    with patch.object(module, "setup", setup), patch.object(module, "act", act):
        candidate, reference = sample(False), sample(True)
        assert candidate == sample(False)
        assert candidate == {k: v for k, v in reference.items() if k != 0}
        assert candidate[1] != candidate[2]
    after = np.random.get_state()
    np.testing.assert_array_equal(before_np[1], after[1])
    assert before_np[0] == after[0] and before_np[2:] == after[2:]
    assert random.getstate() == before_py
