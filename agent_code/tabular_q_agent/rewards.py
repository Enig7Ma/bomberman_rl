"""The rewards the table learns from (plan ``dev/tabiular_q-learning.md`` §5.7).

A transition's reward has three parts:

- **base**: the engine's own score delta, ``REWARD_COIN`` per
  ``COIN_COLLECTED`` and ``REWARD_KILL`` per ``KILLED_OPPONENT``, counting every
  occurrence. It is exactly what the tournament totals, so summed over a round
  it must equal the agent's final score (the Q4 engine tests check this).
- **aids** (objective-changing, off by default): ``crate_aid`` per
  ``CRATE_DESTROYED`` and ``death_aid`` once per death. A suicide reports both
  ``KILLED_SELF`` and ``GOT_KILLED``, which still counts as one death. Bomb
  drops, waiting and invalid moves are never rewarded: bombing for reward is an
  easy exploit, waiting is often right, and invalid moves are mostly lost races
  for a tile.
- **shaping**: ``gamma * phi(s') - phi(s)`` with the coin potential
  ``phi = coin_potential / (1 + d)``, ``d`` the walking distance to the nearest
  visible coin and ``phi = 0`` when there is none or ``s'`` is terminal. As a
  function of the full observation it is a true potential of the game, so it
  does not change which policies are optimal (Ng et al., 1999).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import events as e
import settings


@dataclass(frozen=True)
class Rewards:
    gamma: float
    coin_potential: float = 0.0
    crate_aid: float = 0.0
    death_aid: float = 0.0

    def base(self, events: Sequence[str]) -> float:
        return float(
            settings.REWARD_COIN * events.count(e.COIN_COLLECTED)
            + settings.REWARD_KILL * events.count(e.KILLED_OPPONENT)
        )

    def aids(self, events: Sequence[str]) -> float:
        aid = self.crate_aid * events.count(e.CRATE_DESTROYED)
        if e.GOT_KILLED in events or e.KILLED_SELF in events:
            aid += self.death_aid
        return aid

    def potential(self, coin_distance: int | None) -> float:
        if coin_distance is None:
            return 0.0
        return self.coin_potential / (1 + coin_distance)

    def shaping(self, phi: float, phi_next: float) -> float:
        return self.gamma * phi_next - phi
