"""Reduced-D8 configurations match information, rewards and task schedules."""

from dataclasses import asdict
from pathlib import Path

import numpy as np

from agent_code.dqn_agent.encoder import OneHotE3
from agent_code.dqn_agent.features import Features as DQNFeatures
from agent_code.tabular_q_agent.features import ENCODINGS, Features
from agent_code.tabular_q_agent.qtable import QTable
from docs.experiments.d8_reduced import TableFunction
from docs.experiments.d8_report import hierarchical
from training.config import load_curriculum


def test_courses_match_stage_conditions_and_rewards() -> None:
    table = load_curriculum(Path("docs/experiments/d8_reduced_tabular_q_agent.json"))
    dqn = load_curriculum(Path("docs/experiments/d8_reduced_dqn_agent.json"))
    assert [asdict(s) for s in table.stages] == [asdict(s) for s in dqn.stages]
    assert table.evaluations == dqn.evaluations
    assert table.chunk_rounds == dqn.chunk_rounds == 5
    assert table.params["gamma"] == dqn.params["gamma"] == 0.99
    assert table.params["coin_potential"] == dqn.params["c_coin"] == 0.5
    assert table.params["mask"] == dqn.params["mask"] == "best_tier"
    for key in ("crate_aid", "death_aid"):
        assert table.params[key] == dqn.params[key] == 0
    assert table.params["bomb_aid"] == table.params["spot_potential"] == 0
    assert table.params["teacher_share"] == 0


def test_table_probe_adapter_preserves_exact_canonical_row() -> None:
    table = QTable.zeros(ENCODINGS["E3"])
    f = Features(
        mask=63, coin_dir=1, crate_dir=2, bomb_yield=3, danger=1, opp_dir=4, attack=2
    )
    row = ENCODINGS["E3"].encode(f)
    table.q[row] = np.arange(6, dtype=np.float32)
    x = OneHotE3().encode(DQNFeatures(**asdict(f)))
    np.testing.assert_array_equal(TableFunction(table).values(x), table.q[row])


def test_hierarchical_interval_includes_between_training_variation() -> None:
    result = hierarchical([[0.0] * 25, [10.0] * 25, [20.0] * 25])
    assert result["mean"] == 10.0
    assert result["ci95"] == [0.0, 20.0]
    assert result == hierarchical([[0.0] * 25, [10.0] * 25, [20.0] * 25])
