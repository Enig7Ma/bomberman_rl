"""Control comparison must distinguish absent traces from measured zeros."""

from typing import Any

from docs.experiments.dqn_controls import aggregate, backtracks


def test_backtracks_count_returns_but_not_waiting() -> None:
    assert backtracks([(0, 0), (1, 0), (0, 0), (0, 0), (0, 0)]) == 1
    assert backtracks([(0, 0), (1, 0), (2, 0)]) == 0


def test_aggregation_keeps_missing_metrics_and_all_maps() -> None:
    rows: list[dict[str, Any]] = [
        dict.fromkeys(
            ("coins", "crates", "bombs", "suicides", "deaths", "invalid", "timeouts"),
            0,
        )
        | {"seed": seed, "steps": 400, "wait": 0, "backtracks": None}
        for seed in (500, 501)
    ]
    rows[1]["coins"] = 50
    result = aggregate(rows)
    assert result["per_map_coins"] == [0, 50]
    assert result["coins"]["mean_per_round"] == 25
    assert result["steps"]["total"] == 800
    assert result["backtracks"] is None
    assert result["wait"] == 0
