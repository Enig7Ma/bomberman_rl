"""Post-hoc audit invariants; no game rounds or learning required."""

import numpy as np
import pytest

from agent_code.dqn_agent.encoder import OneHotE3
from agent_code.dqn_agent.features import Features
from agent_code.dqn_agent.probe import Probe
from docs.experiments.dqn_d6_audit import is_focus, loss_window


@pytest.mark.parametrize("code", ["bfs_agent", "rule_based_agent"])
def test_focus_seat_handles_repeated_agent_codes(code: str) -> None:
    assert is_focus(code, code)
    assert is_focus(code + "_0", code)
    assert not is_focus(code + "_1", code)
    assert not is_focus(code + "_2", code)
    assert not is_focus("another_agent", code)


def test_encoder_outputs_and_probe_are_independent_copies() -> None:
    encoder = OneHotE3()
    first = encoder.encode(Features(mask=63, coin_dir=0))
    before = first.copy()
    second = encoder.encode(Features(mask=51, coin_dir=5))
    assert not np.shares_memory(first, second)
    second.fill(0)
    np.testing.assert_array_equal(first, before)
    inputs = np.stack([first, first])
    masks = np.ones((2, 6), dtype=bool)
    probe = Probe(inputs, masks)
    inputs.fill(0)
    masks.fill(False)
    np.testing.assert_array_equal(probe.x[0], before)
    assert probe.masks.all()


def test_round_and_update_weighting_are_explicitly_distinct() -> None:
    rows = [
        {
            "rounds_trained": 1,
            "updates": 1,
            "updates_this_round": 1,
            "mean_loss": 1.0,
            "mean_abs_td": 2.0,
            "mean_grad_norm": 3.0,
        },
        {
            "rounds_trained": 2,
            "updates": 4,
            "updates_this_round": 3,
            "mean_loss": 3.0,
            "mean_abs_td": 4.0,
            "mean_grad_norm": 5.0,
        },
    ]
    window = loss_window(rows)
    assert window["measurements"] == 2 and window["updates"] == 4
    assert window["update_first"] == 1 and window["update_last"] == 4
    assert window["mean_loss"] == 2.0
    assert window["mean_loss_weighted"] == 2.5
