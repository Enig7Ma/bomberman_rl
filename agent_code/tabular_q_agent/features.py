"""Categorical features of an observation, and their encoding as a table index.

Plan: ``dev/tabiular_q-learning.md`` §5.3. Every feature is a small integer
computed from the observation alone -- no hidden coins, no bomb owners:

- ``mask``: bit ``i`` set when ``ACTIONS[i]`` is allowed by the safety mask.
- ``coin_dir``: first step towards the nearest reachable visible coin.
- ``crate_dir``: first step towards the best bombing spot, valued as in
  ``bfs_agent`` (crates a blast would destroy, discounted by distance, crates
  already inside a live blast excluded, spots that cannot be fled excluded).
- ``bomb_yield``: live crates a bomb dropped here would destroy, capped at 3.
- ``danger``: a known blast will cover our cell at some upcoming offset.
- ``opp_dir``: first step towards the nearest cell (within ``hunt_radius``)
  whose blast would cover an opponent where it stands.
- ``attack``: what a bomb dropped here does to the opponents -- nothing,
  pressure (the blast reaches one that can still escape) or a trap (one that
  could escape without the bomb cannot with it).

Directions are first steps on a shortest walk over cells that are free now.
Other agents do not block these walks, as in ``bfs_agent``'s planning: they
will have moved by the time we arrive, and the mask already vets the first
step. When several first steps are equally short, one is picked with the
agent's private RNG, never by neighbour order, so the features carry no
orientation bias -- the board symmetries of plan §5.4 rely on that.

``opp_dir`` has a ``HERE`` value (standing on such a cell already), which the
plan's table omitted: without it, "in position" and "no opponent" coincide.
"""

import hashlib
import json
import math
import random
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Final, Literal

from .config import MaskVariant
from .core.attack import opponent_escapes
from .core.params import Params
from .core.planning import DistanceCache, bomb_spots
from .core.safety import Board, assess_actions, threat_bombs
from .core.world_model import (
    ACTIONS,
    BOMB_POWER,
    BOMB_TIMER,
    Geometry,
    Observation,
    Pos,
    Timeline,
    danger_timeline,
    own_bomb,
    step,
)
from .mask import allowed_actions

# Bump on any change to what a feature means, so stale tables are refused.
FEATURE_VERSION: Final = 1

# Direction values. The four moves come first and in ``ACTIONS`` order, so a
# board symmetry permutes 0..3 and leaves NONE and HERE fixed.
UP: Final = 0
RIGHT: Final = 1
DOWN: Final = 2
LEFT: Final = 3
NONE: Final = 4
HERE: Final = 5
MOVE_ACTIONS: Final[tuple[str, ...]] = ACTIONS[:4]

# ``attack`` values.
NO_ATTACK: Final = 0
PRESSURE: Final = 1
TRAP: Final = 2

MAX_YIELD: Final = 3

# Scores closer than this are ties, broken at random.
_TIE: Final = 1e-9

FieldName = Literal[
    "mask", "coin_dir", "crate_dir", "bomb_yield", "danger", "opp_dir", "attack"
]
RADIX: Final[dict[FieldName, int]] = {
    "mask": 1 << len(ACTIONS),
    "coin_dir": 5,  # moves + NONE
    "crate_dir": 6,  # moves + NONE + HERE
    "bomb_yield": MAX_YIELD + 1,
    "danger": 2,
    "opp_dir": 6,  # moves + NONE + HERE
    "attack": 3,
}


@dataclass(frozen=True)
class Features:
    """One abstract state. Fields an encoding omits keep these defaults."""

    mask: int = 0
    coin_dir: int = NONE
    crate_dir: int = NONE
    bomb_yield: int = 0
    danger: int = 0
    opp_dir: int = NONE
    attack: int = NO_ATTACK

    def values(self) -> dict[FieldName, int]:
        return {
            "mask": self.mask,
            "coin_dir": self.coin_dir,
            "crate_dir": self.crate_dir,
            "bomb_yield": self.bomb_yield,
            "danger": self.danger,
            "opp_dir": self.opp_dir,
            "attack": self.attack,
        }


_DEFAULT_VALUES: Final = Features().values()


@dataclass(frozen=True)
class Encoding:
    """A mixed-radix index over ``fields``, most significant first."""

    name: str
    fields: tuple[FieldName, ...]

    @property
    def radices(self) -> tuple[int, ...]:
        return tuple(RADIX[name] for name in self.fields)

    @property
    def n_states(self) -> int:
        return math.prod(self.radices)

    @property
    def schema_id(self) -> str:
        """Identifies what a table row means; a table saved under another id
        must not be loaded."""
        schema = {
            "version": FEATURE_VERSION,
            "name": self.name,
            "fields": self.fields,
            "radices": self.radices,
            "actions": ACTIONS,
        }
        digest = hashlib.sha256(json.dumps(schema, sort_keys=True).encode())
        return digest.hexdigest()[:16]

    def encode(self, features: Features) -> int:
        values = features.values()
        index = 0
        for name in self.fields:
            value, radix = values[name], RADIX[name]
            if not 0 <= value < radix:
                raise ValueError(f"{name}={value} is outside 0..{radix - 1}")
            index = index * radix + value
        for name, value in values.items():
            if name not in self.fields and value != _DEFAULT_VALUES[name]:
                raise ValueError(f"{self.name} has no field {name}, got {value}")
        return index

    def decode(self, index: int) -> Features:
        if not 0 <= index < self.n_states:
            raise ValueError(f"index {index} is outside 0..{self.n_states - 1}")
        values = dict(_DEFAULT_VALUES)
        for name in reversed(self.fields):
            index, values[name] = divmod(index, RADIX[name])
        return Features(**values)


_E1: Final[tuple[FieldName, ...]] = ("mask", "coin_dir")
_E2: Final[tuple[FieldName, ...]] = (*_E1, "crate_dir", "bomb_yield", "danger")
_E3: Final[tuple[FieldName, ...]] = (*_E2, "opp_dir", "attack")
ENCODINGS: Final[dict[str, Encoding]] = {
    "E1": Encoding("E1", _E1),
    "E2": Encoding("E2", _E2),
    "E3": Encoding("E3", _E3),
}


def mask_bits(actions: Collection[str]) -> int:
    return sum(1 << i for i, action in enumerate(ACTIONS) if action in actions)


def mask_actions(bits: int) -> tuple[str, ...]:
    """The actions set in ``bits``, in ``ACTIONS`` order."""
    return tuple(action for i, action in enumerate(ACTIONS) if bits >> i & 1)


def attack_category(obs: Observation, board: Board, timeline: Timeline) -> int:
    """What a bomb dropped where we stand does to the opponents.

    The same model as ``core.attack.bomb_attack_value`` (the opponent picks any
    first move, everyone else makes way), reduced to its strongest outcome.
    """
    me = obs.me.pos
    armed = timeline.with_bombs(board.geometry, board.crates, [own_bomb(me)])
    blast = board.geometry.blast(me)
    blockers = frozenset({me})
    best = NO_ATTACK
    for other in obs.others:
        x, y = other.pos
        if abs(x - me[0]) + abs(y - me[1]) > BOMB_POWER + BOMB_TIMER:
            continue
        if opponent_escapes(board, armed, other.pos, blockers):
            if other.pos in blast:
                best = PRESSURE
        elif opponent_escapes(board, timeline, other.pos, blockers):
            return TRAP
    return best


@dataclass(frozen=True)
class Extracted:
    """Everything one step derives from an observation."""

    features: Features
    # Allowed actions in ``ACTIONS`` order; never empty.
    allowed: tuple[str, ...]
    # Best safety tier over the legal actions; 0 means no known escape.
    best_tier: int
    # Walking distance to the nearest reachable visible coin, for the coin
    # potential of plan §5.7; None if there is none.
    coin_distance: int | None


class Extractor:
    """Per-agent feature extraction with caches that outlive a step.

    Geometry depends on the stone walls only and is kept for the whole game;
    distance fields are reused until a crate or bomb changes.
    """

    def __init__(
        self, encoding: Encoding, mask: MaskVariant, rng: random.Random
    ) -> None:
        self.encoding = encoding
        self.mask: MaskVariant = mask
        self.rng = rng
        # ``bfs_agent``'s valuation defaults. Deliberately not
        # ``Params.from_env()``: a ``bfs_agent`` sweep must not change features.
        self.params = Params()
        self._wanted = frozenset(encoding.fields)
        self._geometry: Geometry | None = None
        self._distances = DistanceCache()

    def _geometry_for(self, obs: Observation) -> Geometry:
        if self._geometry is None or not self._geometry.matches(obs.field):
            self._geometry = Geometry(obs.field)
        return self._geometry

    def extract(self, obs: Observation) -> Extracted:
        geometry = self._geometry_for(obs)
        board = Board(obs, geometry)
        timeline = danger_timeline(geometry, board.crates, obs.bombs, obs.explosion_map)
        assessments = assess_actions(obs, board, timeline, threat_bombs(obs))
        allowed = allowed_actions(assessments, self.mask)
        cache = self._distances
        cache.sync(board)
        mine = cache.field(obs.me.pos)
        wanted = self._wanted

        coin_dir, coin_distance = self._coin(obs, board, cache, mine)
        crate_dir = NONE
        bomb_yield = 0
        danger = 0
        opp_dir = NONE
        attack = NO_ATTACK
        if "crate_dir" in wanted:
            crate_dir = self._crate(obs, board, cache, mine, timeline)
        if "bomb_yield" in wanted:
            live = board.crates.difference(timeline.crate_open)
            hits = sum(1 for cell in geometry.blast(obs.me.pos) if cell in live)
            bomb_yield = min(hits, MAX_YIELD)
        if "danger" in wanted:
            danger = int(bool(timeline.lethal_offsets(obs.me.pos)))
        if "opp_dir" in wanted:
            opp_dir = self._opponent(obs, board, cache, mine)
        if "attack" in wanted:
            attack = attack_category(obs, board, timeline)

        features = Features(
            mask=mask_bits(allowed),
            coin_dir=coin_dir,
            crate_dir=crate_dir,
            bomb_yield=bomb_yield,
            danger=danger,
            opp_dir=opp_dir,
            attack=attack,
        )
        return Extracted(
            features=features,
            allowed=tuple(allowed),
            best_tier=max(a.tier for a in assessments),
            coin_distance=coin_distance,
        )

    def _coin(
        self,
        obs: Observation,
        board: Board,
        cache: DistanceCache,
        mine: Mapping[Pos, int],
    ) -> tuple[int, int | None]:
        reachable = {coin: mine[coin] for coin in obs.coins if coin in mine}
        if not reachable:
            return NONE, None
        nearest = min(reachable.values())
        targets = {coin: d for coin, d in reachable.items() if d == nearest}
        return self._direction(board, cache, obs.me.pos, targets), nearest

    def _crate(
        self,
        obs: Observation,
        board: Board,
        cache: DistanceCache,
        mine: Mapping[Pos, int],
        timeline: Timeline,
    ) -> int:
        spots = bomb_spots(obs, board, cache, timeline, self.params)
        if not spots:
            return NONE
        discount = self.params.discount
        scores = {spot.pos: spot.value * discount ** mine[spot.pos] for spot in spots}
        best = max(scores.values())
        targets = {
            pos: mine[pos] for pos, score in scores.items() if score >= best - _TIE
        }
        return self._direction(board, cache, obs.me.pos, targets)

    def _opponent(
        self,
        obs: Observation,
        board: Board,
        cache: DistanceCache,
        mine: Mapping[Pos, int],
    ) -> int:
        cells: dict[Pos, int] = {}
        for other in obs.others:
            for cell in board.geometry.blast(other.pos):
                d = mine.get(cell)
                if cell != other.pos and d is not None and d <= self.params.hunt_radius:
                    cells[cell] = d
        if not cells:
            return NONE
        nearest = min(cells.values())
        targets = {cell: d for cell, d in cells.items() if d == nearest}
        return self._direction(board, cache, obs.me.pos, targets)

    def _direction(
        self,
        board: Board,
        cache: DistanceCache,
        me: Pos,
        targets: Mapping[Pos, int],
    ) -> int:
        """A first step on a shortest walk to one of ``targets``.

        ``targets`` maps each target to its distance from ``me``. ``HERE`` if a
        target is ``me``; ties between first steps are broken by the RNG.
        """
        steps: set[int] = set()
        for target, dist in targets.items():
            if dist == 0:
                steps.add(HERE)
                continue
            for direction, move in enumerate(MOVE_ACTIONS):
                nxt = step(me, move)
                if board.geometry.is_wall(nxt) or not board.walkable(nxt):
                    continue
                if cache.field(nxt).get(target) == dist - 1:
                    steps.add(direction)
        if not steps:
            return NONE
        return self.rng.choice(sorted(steps))
