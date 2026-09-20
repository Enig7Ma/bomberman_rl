"""A fixed-priority policy on the agent's own features: the hand-tuned control.

The learned table sees exactly these features and this mask; the heuristic
ranks them by hand instead of learning:

1. a coin: ``WAIT`` on a coin underfoot, else the move ``coin_dir`` names;
2. ``BOMB``, if a bomb here pressures or traps an opponent or destroys a crate;
3. the move ``crate_dir`` names;
4. the move ``opp_dir`` names;
5. otherwise a uniform allowed action.

A step is only taken if the mask allows it. The gap between this control and
the learned table is what learning adds beyond a sensible hand ordering of the
same inputs; the gap to ``policy="random"`` is what it adds at all.
"""

import random
from collections.abc import Sequence

from .core.world_model import BOMB, WAIT
from .features import HERE, MOVE_ACTIONS, PRESSURE, Features


def _move(direction: int, allowed: Sequence[str]) -> str | None:
    if direction >= len(MOVE_ACTIONS):
        return None
    action = MOVE_ACTIONS[direction]
    return action if action in allowed else None


def heuristic_action(
    features: Features, allowed: Sequence[str], rng: random.Random
) -> str:
    """The heuristic's choice among ``allowed`` (real, not canonical, actions)."""
    if not allowed:
        raise ValueError("the action mask is never empty")
    if features.coin_dir == HERE and WAIT in allowed:
        return WAIT
    coin = _move(features.coin_dir, allowed)
    if coin is not None:
        return coin
    worth_a_bomb = features.attack >= PRESSURE or features.bomb_yield >= 1
    if worth_a_bomb and BOMB in allowed:
        return BOMB
    for direction in (features.crate_dir, features.opp_dir):
        step = _move(direction, allowed)
        if step is not None:
            return step
    return rng.choice(list(allowed))
