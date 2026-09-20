"""Masked epsilon-greedy selection and the one-step Q-learning update.

Plan ``dev/tabiular_q-learning.md`` §5.6. Everything here works on canonical
state indices and table columns; ``callbacks`` translates real actions with
``symmetry`` first. ``allowed`` is always the safety mask, and the same mask
is used for exploration, for greedy selection and inside the maximum of the
target -- otherwise the values would describe a policy that is never played.

    delta = r + gamma * max_{b in allowed(s')} Q[s', b] - Q[s, a]
    Q[s, a] += alpha(s, a) * delta
    n[s, a] += 1

with ``alpha(s, a) = max(alpha_min, (1 + n[s, a]) ** -omega)`` and no bootstrap
term when ``s'`` is terminal.

The first update of a state-action therefore sets it straight to its target.
``r`` is whatever the caller passes, shaping term included (plan §5.7).
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass

from .qtable import QTable


@dataclass(frozen=True)
class Selection:
    # Table column of the chosen (canonical) action.
    action: int
    # Chosen by the exploration branch rather than greedily.
    explored: bool
    # No allowed action of this state had been updated yet: the greedy choice
    # was a uniform draw among zeros.
    unseen: bool


class Learner:
    def __init__(
        self,
        table: QTable,
        *,
        gamma: float,
        alpha_omega: float,
        alpha_min: float,
        rng: random.Random,
    ) -> None:
        self.table = table
        self.gamma = gamma
        self.alpha_omega = alpha_omega
        self.alpha_min = alpha_min
        self.rng = rng

    def step_size(self, state: int, action: int) -> float:
        visits = int(self.table.n[state, action])
        return max(self.alpha_min, (1 + visits) ** -self.alpha_omega)

    def value(self, state: int, allowed: Sequence[int]) -> float:
        """``max_{a in allowed} Q[state, a]``."""
        if not allowed:
            raise ValueError("the action mask is never empty")
        row = self.table.q[state]
        return max(float(row[a]) for a in allowed)

    def select(
        self, state: int, allowed: Sequence[int], epsilon: float = 0.0
    ) -> Selection:
        """With probability ``epsilon`` a uniform allowed action, else the
        allowed action with the highest value, ties broken uniformly."""
        top = self.value(state, allowed)
        counts = self.table.n[state]
        unseen = all(int(counts[a]) == 0 for a in allowed)
        if epsilon > 0.0 and self.rng.random() < epsilon:
            return Selection(self.rng.choice(allowed), explored=True, unseen=unseen)
        row = self.table.q[state]
        best = [a for a in allowed if float(row[a]) == top]
        return Selection(self.rng.choice(best), explored=False, unseen=unseen)

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int | None,
        next_allowed: Sequence[int] = (),
    ) -> float:
        """Apply one transition and return its TD error.

        ``next_state`` None marks a terminal transition (death or end of
        round): the target is the reward alone.
        """
        target = reward
        if next_state is not None:
            target += self.gamma * self.value(next_state, next_allowed)
        current = float(self.table.q[state, action])
        delta = target - current
        self.table.q[state, action] = current + self.step_size(state, action) * delta
        self.table.n[state, action] += 1
        return delta
