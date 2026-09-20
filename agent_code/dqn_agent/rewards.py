"""The rewards a transition is worth: engine score, optional aids and shaping.

A transition's reward has three parts:

- **base**: the engine's own score delta, ``REWARD_COIN`` per
  ``COIN_COLLECTED`` and ``REWARD_KILL`` per ``KILLED_OPPONENT``, counting every
  occurrence. It is exactly what the tournament totals, so summed over a round
  it must equal the agent's final score (the engine-driven tests check this).
- **aids** (objective-changing, off by default): ``crate_aid`` per
  ``CRATE_DESTROYED`` and ``death_aid`` once per death. A suicide reports both
  ``KILLED_SELF`` and ``GOT_KILLED``, which still counts as one death. Bomb
  drops, waiting and invalid moves are never rewarded: bombing for reward is an
  easy exploit, waiting is often right, and invalid moves are mostly lost races
  for a tile. The one exception, ``bomb_aid``, is conditional: it is paid per
  live crate a confirmed bomb will destroy, so a useless bomb still earns
  nothing.
- **shaping**: ``gamma * phi(s') - phi(s)`` with the coin potential
  ``phi = coin_potential / (1 + d)``, ``d`` the walking distance to the nearest
  visible coin and ``phi = 0`` when there is none or ``s'`` is terminal. As a
  function of the full observation it is a true potential of the game, so it
  does not change which policies are optimal (Ng et al., 1999). A second
  potential of the same form, on the distance to the best bombing spot
  (``spot_potential``), is off by default.
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
    bomb_aid: float = 0.0
    spot_potential: float = 0.0

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

    def bomb(self, crates: int) -> float:
        """The aid for a confirmed bomb drop that will destroy ``crates``."""
        return self.bomb_aid * crates

    def potential(
        self, coin_distance: int | None, crate_distance: int | None = None
    ) -> float:
        """Coin potential plus bombing-spot potential, both ``c / (1 + d)``.

        The spot potential drops when a bomb books its spot's crates (the next
        spot is farther away), a shaping penalty on ``BOMB`` of about
        ``-spot_potential``. It is meant to be used with ``bomb_aid``, which
        outweighs that penalty for a bomb that destroys crates.
        """
        phi = 0.0
        if coin_distance is not None:
            phi += self.coin_potential / (1 + coin_distance)
        if crate_distance is not None:
            phi += self.spot_potential / (1 + crate_distance)
        return phi

    def shaping(self, phi: float, phi_next: float) -> float:
        return self.gamma * phi_next - phi
