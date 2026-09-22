"""Submission choice follows the frozen rule, never a post-hoc score ranking."""

import pytest

from docs.experiments.d12_summarize import choose


@pytest.mark.parametrize(
    ("dci", "tci", "ds", "ts", "expected"),
    [
        ([2.0, 4.0], [0.0, 1.0], 9, 0, "dqn_stage2_baseline"),
        ([0.0, 1.0], [2.0, 4.0], 0, 9, "tabular_blended"),
        ([0.0, 3.0], [1.0, 4.0], 0, 1, "dqn_stage2_baseline"),
        ([0.0, 3.0], [1.0, 4.0], 1, 0, "tabular_blended"),
        ([0.0, 3.0], [1.0, 4.0], 0, 0, "tabular_blended"),
        ([0.0, 1.0], [1.0, 2.0], 0, 1, "dqn_stage2_baseline"),
    ],
)
def test_submission_rule(
    dci: list[float], tci: list[float], ds: int, ts: int, expected: str
) -> None:
    dqn = {"delta": sum(dci) / 2, "delta_ci": dci, "suicides_total": ds}
    table = {"delta": sum(tci) / 2, "delta_ci": tci, "suicides_total": ts}
    assert choose(dqn, table)[0] == expected
