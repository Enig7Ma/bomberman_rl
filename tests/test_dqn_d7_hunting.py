"""Hunting protocol: masks, fixed opponent streams and continuous epsilon."""

import logging
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

pytest.importorskip("torch")

from agent_code.dqn_agent.encoder import OneHotE3  # noqa: E402
from agent_code.dqn_agent.learner import Learner  # noqa: E402
from agent_code.peaceful_agent import callbacks as peaceful  # noqa: E402
from docs.experiments.dqn_d7_hunting import (  # noqa: E402
    CONFIG,
    evaluate_preset,
    opponent_rngs,
    summarize,
)
from training.config import load_curriculum  # noqa: E402
from training.dqn import stage_config  # noqa: E402


def test_opponent_rng_reproducible_and_preserves_global_state() -> None:
    before_np, before_py = np.random.get_state(), random.getstate()

    def draws() -> list[str]:
        with opponent_rngs(503):
            agent = SimpleNamespace(logger=logging.getLogger("rng-test"))
            cast(Any, peaceful).setup(agent)
            return [str(cast(Any, peaceful).act(agent, {})) for _ in range(30)]

    assert draws() == draws()
    after = np.random.get_state()
    assert before_np[0] == after[0] and before_np[2:] == after[2:]
    np.testing.assert_array_equal(before_np[1], after[1])
    assert before_py == random.getstate()


def test_hunting_budget_mixture_and_epsilon_boundary() -> None:
    course = load_curriculum(CONFIG)
    assert sum(s.transitions or 0 for s in course.stages) == 400000
    configs = [stage_config(course, s, i + 2, 0) for i, s in enumerate(course.stages)]
    for i, stage in enumerate(course.stages):
        assert (
            stage.lineups[0].opponents
            == (("peaceful_agent" if i == 0 else "coin_collector_agent"),) * 3
        )
        assert [x.weight for x in stage.lineups] == [0.8, 0.08, 0.08, 0.04]
    learners = [Learner(OneHotE3(), c) for c in configs]
    assert learners[0].epsilon(0, 200000) == 0.3
    assert learners[0].epsilon(200000, 200000) == pytest.approx(
        learners[1].epsilon(0, 200000)
    )
    assert learners[1].epsilon(40000, 200000) == pytest.approx(0.05)


def test_pooled_forced_fraction_is_action_weighted() -> None:
    keys = (
        "score",
        "kills",
        "coins",
        "crates",
        "suicides",
        "survived",
        "bombs",
        "invalid",
        "timeouts",
        "wait",
        "forced",
        "empty_bombs",
        "empty_contexts",
    )
    rows: list[dict[str, Any]] = [
        dict.fromkeys(keys, 0) | {"actions": a, "steps": a} for a in (10, 100)
    ]
    rows[0]["forced"] = 10
    assert summarize(rows)["forced_step_fraction"] == pytest.approx(10 / 110)


def test_real_evaluation_records_engine_credit_and_actions(tmp_path: Path) -> None:
    from agent_code.dqn_agent.network import QNetwork

    model = tmp_path / "q_net.npz"
    QNetwork.random(OneHotE3(), 0).save(model)
    result = evaluate_preset(model, tmp_path, "vs-peaceful", "random", [599])
    row = result["per_map"][0]
    assert row["score"] == row["coins"] + 5 * row["kills"]
    assert row["actions"] == row["steps"]
    assert 0 <= result["forced_step_fraction"] <= 1
