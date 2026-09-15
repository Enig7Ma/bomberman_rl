"""Board symmetries: canonical states and the matching action mapping.

Plan ``dev/tabiular_q-learning.md`` §5.4. The stone walls, the start corners,
the crate distribution and the engine's rules are all invariant under the 8
rotations and reflections of the square board (the dihedral group D4). A
situation and its mirror image deserve the same action values with the
actions mirrored, so the Q-table stores one representative per orbit:

    index, g = canonical(features, encoding)   # g maps the real board onto it
    a_canonical = to_canonical(a, g)           # for learning and lookup
    a = from_canonical(a_canonical, g)         # to play the table's choice

Conventions, in the GUI view (``x`` to the right, ``y`` down): a ``Symmetry``
first mirrors left-right (``x -> n - x``) if ``mirrored``, then turns
``rotations`` quarter turns clockwise (``(x, y) -> (n - y, x)``), with
``n = size - 1``. On directions a clockwise turn maps UP -> RIGHT -> DOWN ->
LEFT -> UP and the mirror swaps LEFT and RIGHT; ``NONE``, ``HERE``, ``WAIT``
and ``BOMB`` are fixed.

The representative is the image with the smallest index, taking the first
symmetry in ``SYMMETRIES`` (identity first) on equal indices. A state fixed by
several symmetries therefore always uses the same one. That is consistent, but
it does not force equal values for actions its stabiliser swaps; the plan
accepts this.

All of this is sound only if the features are equivariant: extracting from a
transformed game state must give the transformed features.
``transform_game_state`` exists so the tests can check exactly that on real
engine states.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from .core.world_model import Pos
from .features import MOVE_ACTIONS, Encoding, Features

_MOVES: Final = len(MOVE_ACTIONS)
# Left-right mirror on direction values UP, RIGHT, DOWN, LEFT.
_MIRROR: Final = (0, 3, 2, 1)
# Mask bits of WAIT and BOMB, which no symmetry moves.
_FIXED_BITS: Final = ~((1 << _MOVES) - 1)


@dataclass(frozen=True)
class Symmetry:
    """One element of D4; see the module docstring for the conventions."""

    rotations: int
    mirrored: bool

    def direction(self, value: int) -> int:
        """Image of a direction value; NONE and HERE are fixed."""
        if value >= _MOVES:
            return value
        if self.mirrored:
            value = _MIRROR[value]
        return (value + self.rotations) % _MOVES

    def action(self, action: str) -> str:
        """Image of an action; WAIT and BOMB are fixed."""
        if action not in MOVE_ACTIONS:
            return action
        return MOVE_ACTIONS[self.direction(MOVE_ACTIONS.index(action))]

    def cell(self, pos: Pos, size: int) -> Pos:
        """Image of a cell on a ``size`` x ``size`` board."""
        n = size - 1
        x, y = pos
        if self.mirrored:
            x = n - x
        for _ in range(self.rotations):
            x, y = n - y, x
        return x, y


SYMMETRIES: Final[tuple[Symmetry, ...]] = tuple(
    Symmetry(rotations, mirrored)
    for mirrored in (False, True)
    for rotations in range(4)
)
IDENTITY: Final = SYMMETRIES[0]


def _key(symmetry: Symmetry) -> tuple[int, ...]:
    # D4 acts faithfully on the four directions, so this identifies an element.
    return tuple(symmetry.direction(d) for d in range(_MOVES))


_BY_KEY: Final[dict[tuple[int, ...], Symmetry]] = {_key(s): s for s in SYMMETRIES}


def compose(outer: Symmetry, inner: Symmetry) -> Symmetry:
    """The symmetry that applies ``inner`` first, then ``outer``."""
    return _BY_KEY[tuple(outer.direction(inner.direction(d)) for d in range(_MOVES))]


def inverse(symmetry: Symmetry) -> Symmetry:
    return next(s for s in SYMMETRIES if compose(s, symmetry) == IDENTITY)


def transform_mask(bits: int, symmetry: Symmetry) -> int:
    """Image of an action mask: move bits permuted, WAIT and BOMB kept."""
    out = bits & _FIXED_BITS
    for d in range(_MOVES):
        if bits >> d & 1:
            out |= 1 << symmetry.direction(d)
    return out


def transform_features(features: Features, symmetry: Symmetry) -> Features:
    return replace(
        features,
        mask=transform_mask(features.mask, symmetry),
        coin_dir=symmetry.direction(features.coin_dir),
        crate_dir=symmetry.direction(features.crate_dir),
        opp_dir=symmetry.direction(features.opp_dir),
    )


def canonical(features: Features, encoding: Encoding) -> tuple[int, Symmetry]:
    """The table index of the orbit's representative, and the symmetry that
    maps ``features`` onto it."""
    best_index = encoding.encode(features)
    best = IDENTITY
    for symmetry in SYMMETRIES[1:]:
        index = encoding.encode(transform_features(features, symmetry))
        if index < best_index:
            best_index, best = index, symmetry
    return best_index, best


def to_canonical(action: str, symmetry: Symmetry) -> str:
    """The action in the canonical state that corresponds to ``action``."""
    return symmetry.action(action)


def from_canonical(action: str, symmetry: Symmetry) -> str:
    """The real action that corresponds to the canonical state's ``action``."""
    return inverse(symmetry).action(action)


def transform_grid[T: np.generic](grid: NDArray[T], symmetry: Symmetry) -> NDArray[T]:
    """A board-shaped array with every cell moved to its image."""
    width, height = grid.shape
    if width != height:
        raise ValueError(f"board symmetries need a square board, got {grid.shape}")
    n = width - 1
    xs, ys = np.indices(grid.shape)
    if symmetry.mirrored:
        xs = n - xs
    for _ in range(symmetry.rotations):
        xs, ys = n - ys, xs
    out = np.empty_like(grid)
    out[xs, ys] = grid
    return out


def transform_game_state(
    state: Mapping[str, Any], symmetry: Symmetry
) -> dict[str, Any]:
    """The ``game_state`` dict of the transformed board.

    Moves the field, the explosion map and every position; everything else
    (round, step, names, scores, bomb timers) is copied unchanged.
    """
    field: NDArray[Any] = np.asarray(state["field"])
    size = int(field.shape[0])

    def move(xy: Sequence[Any]) -> Pos:
        return symmetry.cell((int(xy[0]), int(xy[1])), size)

    def player(entry: Sequence[Any]) -> tuple[Any, Any, Any, Pos]:
        name, score, can_bomb, xy = entry
        return name, score, can_bomb, move(xy)

    transformed = dict(state)
    transformed["field"] = transform_grid(field, symmetry)
    transformed["explosion_map"] = transform_grid(
        np.asarray(state["explosion_map"]), symmetry
    )
    transformed["self"] = player(state["self"])
    transformed["others"] = [player(other) for other in state["others"]]
    transformed["bombs"] = [(move(xy), timer) for xy, timer in state["bombs"]]
    transformed["coins"] = [move(xy) for xy in state["coins"]]
    return transformed
