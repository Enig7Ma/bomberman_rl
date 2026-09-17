"""D6 experimental protocol: fixed probes, complete outcomes, strict inference."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from docs.experiments.dqn_d6 import (
    ENCODER,
    aggregate,
    checksum,
    collect_probe,
    dump,
    run,
    select,
)
from tournament.results import AgentRoundResult, RoundResult


def result(coins: int, steps: int, *, died: bool = False) -> RoundResult:
    agent = AgentRoundResult(
        "dqn_agent",
        "dqn_agent",
        0,
        coins,
        coins,
        0,
        0,
        0,
        0,
        0,
        steps,
        steps,
        not died,
        0.0,
        0.0,
        0.0,
        0,
    )
    return RoundResult(
        "coin-heaven", 500, ("dqn_agent",), "candidate", 0, steps, (agent,)
    )


def test_failed_rounds_remain_in_denominator_and_steps() -> None:
    summary = aggregate([result(50, 120), result(30, 400), result(0, 10, died=True)])
    assert summary["mean_coins"] == pytest.approx(80 / 3)
    assert summary["all50_fraction"] == pytest.approx(1 / 3)
    assert summary["mean_steps_success"] == 120
    assert summary["mean_steps_all"] == pytest.approx(530 / 3)
    assert summary["deaths"] == 1 and not summary["strict_126"]
    assert aggregate([result(0, 400)])["mean_steps_success"] is None
    assert not aggregate([result(50, 120), result(50, 127)])["strict_126"]


def test_selection_excludes_unstable_and_uses_validation_not_loss(
    tmp_path: Path,
) -> None:
    paths = [tmp_path / str(i) for i in range(4)]
    for i, (stable, coins, steps) in enumerate(
        [(False, 50, 110), (True, 49, 110), (True, 50, 126), (True, 50, 130)]
    ):
        dump(
            paths[i] / "run_summary.json",
            {
                "directory": str(paths[i]),
                "stable": stable,
                "loss_last20": 100 if i == 2 else 0.0001,
                "evaluation": [
                    {
                        "mean_coins": coins,
                        "all50_fraction": coins == 50,
                        "mean_steps_all": steps,
                    }
                ],
            },
        )
    assert select(paths)["selected"] == str(paths[2])


def test_probe_collection_reproduces_exactly_without_training(tmp_path: Path) -> None:
    a, b = collect_probe(tmp_path / "a"), collect_probe(tmp_path / "b")
    assert checksum(a) == checksum(b)
    with np.load(a, allow_pickle=False) as data:
        assert data["x"].shape == (2000, 32)
        assert data["masks"].shape == (2000, 6)
        assert data["masks"].any(axis=1).all()
        assert str(data["schema_id"].item()) == ENCODER.schema_id
    manifest = json.loads((a.parent / "probe_manifest.json").read_text())
    assert {r["seed"] for r in manifest["selected_rows"]} == set(range(900, 910))
    assert {r["agent"] for r in manifest["selected_rows"]} == {
        "bfs_agent",
        "rule_based_agent",
    }
    assert not list(tmp_path.rglob("replay.npz"))
    assert not list(tmp_path.rglob("checkpoint.pt"))


@pytest.mark.parametrize("unstable", [False, True])
def test_actual_save_hook_measures_initial_snapshot_and_stops_bad_q(
    tmp_path: Path, unstable: bool
) -> None:
    pytest.importorskip("torch")
    probe = tmp_path / "probe.npz"
    x = np.full((2, 32), 1e9 if unstable else 0, dtype=np.float32)
    np.savez_compressed(
        probe,
        x=x,
        masks=np.ones((2, 6), dtype=bool),
        schema_id=np.array(ENCODER.schema_id),
    )
    cfg: dict[str, Any] = {
        "name": "test-d6",
        "agent": "dqn_agent",
        "params": {"probe_path": str(probe), "replay_size": 1000},
        "chunk_rounds": 1,
        "eval_every_transitions": 10000,
        "evaluations": [],
        "stages": [
            {
                "name": "test",
                "scenario": "coin-heaven",
                "lineups": [{"opponents": []}],
                "transitions": 1,
                "epsilon_start": 0.3,
                "epsilon_end": 0.05,
            }
        ],
    }
    path = tmp_path / "config.json"
    dump(path, cfg)
    out = tmp_path / "run"
    summary = run(path, out, 0)
    if unstable:
        assert "D-S7" in summary["failure"]
        assert summary["transitions"] == 0
        assert (out / "failure.txt").exists()
        assert not (out / "checkpoint.pt").exists()
    else:
        assert summary["failure"] is None
        saved = (out / "run_summary.json").read_bytes()
        assert run(path, out, 0) == summary
        assert (out / "run_summary.json").read_bytes() == saved
        with pytest.raises(ValueError, match="another config or seed"):
            run(path, out, 1)
        assert len(summary["evaluation"]) == 2
        assert summary["evaluation"][0]["transitions"] == 0
        probes = list(out.glob("checkpoints/*/probe.json"))
        assert len(probes) == 2
        records = [json.loads(p.read_text()) for p in probes]
        assert records[0]["policy_churn"] is None
        assert all(r["max_abs_q"] <= 50 for r in records)
        assert (out / "checkpoints" / "transition_000000000" / "replay.npz").exists()
