"""Transition budgets preserve legacy courses and commit table/cursor together."""

import copy
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from agent_code.tabular_q_agent.features import ENCODINGS
from agent_code.tabular_q_agent.learner import Learner, Selection
from agent_code.tabular_q_agent.metrics import read_records
from agent_code.tabular_q_agent.qtable import QTable
from training.config import CurriculumError, parse_curriculum
from training.driver import read_chunk_records, snapshots, train_run
from training.evaluate import evaluate_run
from training.tabular_steps import publish, replace_derived


def course() -> dict[str, Any]:
    return {
        "name": "transition-test",
        "agent": "tabular_q_agent",
        "params": {"encoding": "E1", "gamma": 0.99},
        "chunk_rounds": 1,
        "eval_every_transitions": 300,
        "evaluations": [{"preset": "coin-heaven-solo", "seeds": 1}],
        "stages": [
            {
                "name": "coins",
                "scenario": "coin-heaven",
                "lineups": [{"opponents": []}],
                "transitions": 401,
                "epsilon_start": 0.3,
                "epsilon_end": 0.05,
            }
        ],
    }


def test_windows_derived_file_lock_retries_and_preserves_old_file(
    tmp_path: Path,
) -> None:
    source, target = tmp_path / "metrics.tmp", tmp_path / "metrics.jsonl"
    source.write_text("new")
    target.write_text("old")
    original = Path.replace
    attempts = 0

    def locked(path: Path, destination: Path) -> Path:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            assert target.read_text() == "old"
            raise PermissionError("sharing lock")
        return original(path, destination)

    with (
        patch.object(Path, "replace", locked),
        patch("training.tabular_steps.time.sleep"),
    ):
        replace_derived(source, target)
    assert attempts == 3 and target.read_text() == "new"
    source.write_text("later")
    with (
        patch.object(
            Path, "replace", side_effect=PermissionError("persistent")
        ) as fail,
        patch("training.tabular_steps.time.sleep"),
    ):
        with pytest.raises(PermissionError):
            replace_derived(source, target)
    assert fail.call_count == 5
    assert target.read_text() == "new" and source.read_text() == "later"


def test_transition_parser_schedule_and_mixed_budget_rejection() -> None:
    c = parse_curriculum(course())
    assert parse_curriculum(asdict(c)) == c
    s = c.stages[0]
    assert s.epsilon_at_transition(0) == 0.3
    assert s.epsilon_at_transition(120) == pytest.approx(0.3 - 0.25 * 120 / 240.6)
    assert s.epsilon_at_transition(401) == 0.05
    raw = course()
    second = copy.deepcopy(raw["stages"][0])
    second.pop("transitions")
    second.update(name="round-mode", rounds=1)
    raw["stages"].append(second)
    with pytest.raises(CurriculumError, match="cannot mix"):
        parse_curriculum(raw)


def test_budget_snapshot_and_noop_resume(tmp_path: Path) -> None:
    c = parse_curriculum(course())
    used: list[float] = []
    original = Learner.select

    def select(
        learner: Learner, state: int, allowed: Sequence[int], epsilon: float
    ) -> Selection:
        used.append(epsilon)
        return original(learner, state, allowed, epsilon)

    with patch.object(Learner, "select", select):
        rows = train_run(c, tmp_path, 0)
    assert 401 <= rows[-1].total_transitions <= 800
    assert sum(r.transitions for r in rows) == rows[-1].total_transitions
    assert rows[-1].epsilon == 0.05
    assert len(used) == rows[-1].total_transitions
    for i, epsilon in enumerate(used):
        assert epsilon == pytest.approx(c.stages[0].epsilon_at_transition(i))
    saved = QTable.load(tmp_path / "q_table.npz", ENCODINGS["E1"])
    assert saved.meta["transition_driver"]["next_stage"] == 1
    assert snapshots(tmp_path)[-1][0] == rows[-1].total_transitions
    before = (tmp_path / "q_table.npz").read_bytes()
    assert train_run(c, tmp_path, 0) == []
    assert (tmp_path / "q_table.npz").read_bytes() == before
    metrics = [
        json.loads(s) for s in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ]
    assert sum(r["steps"] for r in metrics) == saved.meta["steps_trained"]
    assert len(read_records(tmp_path / "metrics.jsonl")) == len(metrics)


@pytest.mark.parametrize("after_save", [False, True])
def test_interrupted_chunk_resume_preserves_table_rng_and_cursor(
    tmp_path: Path, after_save: bool
) -> None:
    c = parse_curriculum(course())
    full, split = tmp_path / "full", tmp_path / "split"
    train_run(c, full, 0)
    original_save = QTable.save

    def fail_save(table: QTable, path: Path) -> None:
        if table.meta.get("transition_driver", {}).get("records"):
            raise OSError("interrupted before commit")
        original_save(table, path)

    def fail_publish(run_dir: Path, table: QTable) -> None:
        if table.meta["transition_driver"]["records"]:
            raise OSError("interrupted after commit")
        publish(run_dir, table)

    target = (
        "training.tabular_steps.publish"
        if after_save
        else "agent_code.tabular_q_agent.qtable.QTable.save"
    )
    with patch(target, fail_publish if after_save else fail_save):
        with pytest.raises(OSError, match="interrupted"):
            train_run(c, split, 0)
    # Uncommitted/partial metric tails must not enter resumed counters.
    with (split / "metrics.jsonl").open("a") as f:
        f.write('{"partial":')
    train_run(c, split, 0)
    a = QTable.load(full / "q_table.npz", ENCODINGS["E1"])
    b = QTable.load(split / "q_table.npz", ENCODINGS["E1"])
    np.testing.assert_array_equal(a.q, b.q)
    np.testing.assert_array_equal(a.n, b.n)
    assert a.meta["transition_driver"]["rng"] == b.meta["transition_driver"]["rng"]
    assert a.meta["steps_trained"] == b.meta["steps_trained"]
    assert len(read_chunk_records(full)) == len(read_chunk_records(split))


def test_stage_change_and_evaluation_use_transition_axis(tmp_path: Path) -> None:
    raw = course()
    raw["stages"][0]["transitions"] = 1
    second = copy.deepcopy(raw["stages"][0])
    second.update(name="crates", scenario="loot-crate", epsilon_start=0.2)
    raw["stages"].append(second)
    c = parse_curriculum(raw)
    rows = train_run(c, tmp_path, 0)
    assert [r.stage for r in rows] == ["coins", "crates"]
    table = QTable.load(tmp_path / "q_table.npz", ENCODINGS["E1"])
    assert table.meta["transition_driver"]["next_stage"] == 2
    assert table.meta["steps_trained"] == sum(r.transitions for r in rows)
    before = (tmp_path / "q_table.npz").read_bytes()
    points = evaluate_run(tmp_path)["coin-heaven-solo"]
    assert [p.transitions for p in points] == [r.total_transitions for r in rows]
    assert (tmp_path / "q_table.npz").read_bytes() == before
    assert len(list((tmp_path / "eval").glob("*_transition_*.jsonl"))) == 2
