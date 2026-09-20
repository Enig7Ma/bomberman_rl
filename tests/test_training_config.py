"""Tests for curriculum parsing and validation (``training.config``)."""

import copy
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from training.config import (
    CurriculumError,
    Lineup,
    load_curriculum,
    parse_curriculum,
)

VALID: dict[str, Any] = {
    "name": "two-stages",
    "params": {"encoding": "E1"},
    "chunk_rounds": 2,
    "eval_every": 4,
    "evaluations": [{"preset": "coin-heaven-solo", "seeds": 2, "seed_start": 500}],
    "stages": [
        {
            "name": "coins",
            "scenario": "coin-heaven",
            "lineups": [{"opponents": [], "weight": 1.0}],
            "rounds": 4,
            "epsilon_start": 0.3,
            "epsilon_end": 0.05,
            "decay_share": 0.5,
            "replay_share": 0.0,
        },
        {
            "name": "crates",
            "scenario": "loot-crate",
            "lineups": [
                {"opponents": ["peaceful_agent"], "weight": 1},
                {"opponents": ["frozen", "rule_based_agent"], "weight": 2.5},
                {"opponents": [], "scenario": "classic"},
            ],
            "rounds": 6,
            "epsilon_start": 0.2,
            "epsilon_end": 0.2,
            "replay_share": 0.5,
        },
    ],
}

DELETE = object()


def changed(path: Sequence[str | int], value: object) -> dict[str, Any]:
    raw = copy.deepcopy(VALID)
    target: Any = raw
    for key in path[:-1]:
        target = target[key]
    if value is DELETE:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return raw


def test_a_valid_curriculum_parses() -> None:
    curriculum = parse_curriculum(VALID)
    assert curriculum.total_rounds == 10
    assert curriculum.agent_config().encoding == "E1"
    coins, crates = curriculum.stages
    assert coins.lineups == (Lineup((), 1.0),)
    assert crates.lineups[1] == Lineup(("frozen", "rule_based_agent"), 2.5)
    assert crates.lineups[2] == Lineup((), 1.0, "classic")
    assert coins.lineups[0].scenario is None
    assert crates.decay_share == 0.6  # default
    assert curriculum.evaluations[0].preset == "coin-heaven-solo"


def test_the_stored_form_parses_back_unchanged(tmp_path: Path) -> None:
    curriculum = parse_curriculum(VALID)
    path = tmp_path / "curriculum.json"
    path.write_text(json.dumps(asdict(curriculum)))
    assert load_curriculum(path) == curriculum


def test_epsilon_decays_over_the_decay_share() -> None:
    coins = parse_curriculum(VALID).stages[0]  # 4 rounds, decay over 2
    assert coins.epsilon_at(0) == 0.3
    assert coins.epsilon_at(1) == pytest.approx(0.175)
    assert coins.epsilon_at(2) == 0.05
    assert coins.epsilon_at(3) == 0.05


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (["extra"], 1),
        (["name"], DELETE),
        (["chunk_rounds"], 0),
        (["eval_every"], 3),
        (["eval_every"], 1),
        (["params"], {"epsilon": 0.2}),
        (["params"], {"seed": 1}),
        (["params"], {"encoding": "E4"}),
        (["params"], {"temperature": 1}),
        (["evaluations", 0, "preset"], "vs-nobody"),
        (["evaluations", 0, "seeds"], 0),
        (["stages"], []),
        (["stages", 0, "extra"], True),
        (["stages", 0, "scenario"], "moon"),
        (["stages", 0, "rounds"], 3),
        (["stages", 0, "lineups"], []),
        (["stages", 0, "replay_share"], 0.1),
        (["stages", 0, "epsilon_start"], 1.5),
        (["stages", 1, "name"], "coins"),
        (["stages", 1, "replay_share"], 1.0),
        (["stages", 1, "lineups", 0, "weight"], 0),
        (["stages", 1, "lineups", 0, "opponents"], ["no_such_agent"]),
        (["stages", 1, "lineups", 0, "opponents"], ["tabular_q_agent"]),
        (["stages", 1, "lineups", 0, "opponents"], ["tabular_frozen_s0"]),
        (["stages", 1, "lineups", 0, "opponents"], ["peaceful_agent"] * 4),
        (["stages", 1, "lineups", 2, "scenario"], "moon"),
    ],
)
def test_invalid_curricula_are_rejected(path: list[str | int], value: object) -> None:
    with pytest.raises(CurriculumError):
        parse_curriculum(changed(path, value))


def test_a_file_that_is_not_json_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{ not json")
    with pytest.raises(CurriculumError):
        load_curriculum(path)
