"""Diagnostic counting must not confuse opportunities, explosions and outcomes."""

import json
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from agent_code.dqn_agent.core.world_model import Observation
from docs.experiments.dqn_d7_diagnose import reachable_coins, summarize


def test_bomb_denominators_and_pending_bombs() -> None:
    trace: list[dict[str, Any]] = []
    for action, productive, allowed in (
        ("BOMB", True, True),
        ("WAIT", True, False),
        ("BOMB", False, True),
    ):
        trace.append(
            {
                "action": action,
                "features": {"bomb_yield": int(productive), "attack": 0},
                "allowed": ["WAIT", "BOMB"] if allowed else ["WAIT"],
                "pos": [1, 1],
                "reachable_coins": 0,
                "score_after": 0,
            }
        )
    result = summarize(trace, [{"crates": 2}])
    assert result["productive_bomb_opportunities"] == 1
    assert result["productive_bomb_choices"] == 1
    assert result["bomb_allowed"] == 2
    assert result["crate_destroying_bombs"] == 1
    assert result["pending_bombs"] == 1
    assert result["empty_bombs"] == result["empty_contexts"] == 1
    assert result["immediate_backtracks"] == 0
    json.dumps(result, allow_nan=False)


def test_reachability_uses_only_visible_coins_and_blocks_bombs() -> None:
    field = np.full((5, 5), -1, dtype=np.int64)
    field[1:4, 1] = 0
    obs = cast(
        Observation,
        SimpleNamespace(
            field=field,
            me=SimpleNamespace(pos=(1, 1)),
            coins=((3, 1),),
            bombs=(((2, 1), 3),),
        ),
    )
    assert reachable_coins(obs) == 0
    cast(Any, obs).bombs = ()
    assert reachable_coins(obs) == 1
    cast(Any, obs).coins = ()
    assert reachable_coins(obs) == 0
