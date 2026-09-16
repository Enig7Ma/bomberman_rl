"""Typed view of the observation, blast geometry and the danger forecast.

The framework hands ``act`` an untyped dict whose coordinates are a mixture of
Python and NumPy integers. ``Observation.from_game_state`` is the single place
that dict is read; everything downstream works on plain tuples of ``int``.

Timing convention used throughout the agent: *offset* ``t`` counts action
resolutions from the observation being handled, so the action ``act`` returns
now resolves at offset 0. After each resolution the engine ages explosions,
detonates bombs, then kills every agent standing in a dangerous blast. "Lethal
at offset ``t``" means an agent standing on the cell after action ``t`` dies.

Engine facts encoded here (``environment.py``; checked by the differential
test in ``tests/test_bfs_agent_world_model.py``, not read off the rules PDF):

- A bomb observed with timer ``k`` detonates after action ``k``; its blast is
  lethal at offsets ``k .. k + EXPLOSION_TIMER - 1`` -- two death checks.
- Its tile blocks movement for the actions at offsets ``0 .. k``.
- Crates in the blast vanish after action ``k``: enterable from ``k + 1``.
- ``explosion_map[c] == v`` with ``v >= 1`` is lethal at offsets ``0 .. v-1``.
  ``v == 0`` is harmless, not a hidden next-step hazard.
- A bomb dropped by the current action detonates after action ``BOMB_TIMER``.
- Blasts stop at stone walls, pass through crates and never chain-detonate, so
  a blast's cells depend on the walls alone and are cached per position.
"""

from collections.abc import Container, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

import settings

Pos = tuple[int, int]

BOMB_POWER: Final[int] = settings.BOMB_POWER
BOMB_TIMER: Final[int] = settings.BOMB_TIMER
EXPLOSION_TIMER: Final[int] = settings.EXPLOSION_TIMER

WALL: Final = -1
FREE: Final = 0
CRATE: Final = 1

UP: Final = "UP"
RIGHT: Final = "RIGHT"
DOWN: Final = "DOWN"
LEFT: Final = "LEFT"
WAIT: Final = "WAIT"
BOMB: Final = "BOMB"

MOVES: Final[dict[str, Pos]] = {UP: (0, -1), RIGHT: (1, 0), DOWN: (0, 1), LEFT: (-1, 0)}
ACTIONS: Final = (UP, RIGHT, DOWN, LEFT, WAIT, BOMB)


def step(pos: Pos, action: str) -> Pos:
    """Where ``action`` would take an agent at ``pos`` if the move succeeds."""
    dx, dy = MOVES.get(action, (0, 0))
    return pos[0] + dx, pos[1] + dy


@dataclass(frozen=True)
class Player:
    """One agent as seen in the observation."""

    name: str
    score: int
    can_bomb: bool
    pos: Pos


@dataclass(frozen=True)
class Observation:
    """The game state handed to ``act``, with every coordinate a plain ``int``."""

    round: int
    step: int
    field: NDArray[np.int64]
    me: Player
    others: tuple[Player, ...]
    bombs: tuple[tuple[Pos, int], ...]
    coins: tuple[Pos, ...]
    explosion_map: NDArray[np.float64]

    @classmethod
    def from_game_state(cls, state: Mapping[str, Any]) -> "Observation":
        return cls(
            round=int(state["round"]),
            step=int(state["step"]),
            field=np.asarray(state["field"], dtype=np.int64),
            me=_player(state["self"]),
            others=tuple(_player(other) for other in state["others"]),
            bombs=tuple((_pos(xy), int(timer)) for xy, timer in state["bombs"]),
            coins=tuple(_pos(xy) for xy in state["coins"]),
            explosion_map=np.asarray(state["explosion_map"], dtype=np.float64),
        )


def _pos(xy: Sequence[Any]) -> Pos:
    return int(xy[0]), int(xy[1])


def _player(entry: Sequence[Any]) -> Player:
    name, score, can_bomb, xy = entry
    return Player(
        name=str(name), score=int(score), can_bomb=bool(can_bomb), pos=_pos(xy)
    )


class Geometry:
    """Everything that depends on the stone walls alone.

    Stone walls are identical in every round and never destroyed, so neighbour
    lists are computed once and kept for the whole game. ``matches`` lets the
    caller detect a different board (tests, or a changed ``settings.py``).
    """

    def __init__(self, field: NDArray[np.int64]) -> None:
        self.walls: NDArray[np.bool_] = field == WALL
        width, height = field.shape
        self.width = width
        self.height = height
        wall_rows: list[list[bool]] = self.walls.tolist()
        self._wall_rows = wall_rows
        self._blasts: dict[Pos, tuple[Pos, ...]] = {}
        self.neighbors: dict[Pos, tuple[Pos, ...]] = {}
        for x in range(width):
            for y in range(height):
                if wall_rows[x][y]:
                    continue
                self.neighbors[(x, y)] = tuple(
                    (x + dx, y + dy)
                    for dx, dy in MOVES.values()
                    if 0 <= x + dx < width
                    and 0 <= y + dy < height
                    and not wall_rows[x + dx][y + dy]
                )

    def matches(self, field: NDArray[np.int64]) -> bool:
        return field.shape == self.walls.shape and bool(
            np.array_equal(field == WALL, self.walls)
        )

    def is_wall(self, pos: Pos) -> bool:
        x, y = pos
        if not (0 <= x < self.width and 0 <= y < self.height):
            return True
        return self._wall_rows[x][y]

    def blast(self, pos: Pos) -> tuple[Pos, ...]:
        """Cells a bomb at ``pos`` hits, in the engine's order.

        Mirrors ``items.Bomb.get_blast_coords``: up to ``BOMB_POWER`` cells in
        each direction, stopping at the first stone wall. Crates do not stop a
        blast, so the result depends on the walls only and is cached forever.
        """
        cached = self._blasts.get(pos)
        if cached is None:
            x, y = pos
            cells = [pos]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                for reach in range(1, BOMB_POWER + 1):
                    cell = (x + dx * reach, y + dy * reach)
                    if self.is_wall(cell):
                        break
                    cells.append(cell)
            cached = tuple(cells)
            self._blasts[pos] = cached
        return cached


def legal_actions(obs: Observation) -> list[str]:
    """Actions the engine would execute rather than record as invalid.

    A move needs a free destination: no wall, crate, bomb or other agent.
    ``WAIT`` is always legal, ``BOMB`` only while the agent holds its bomb.
    """
    blocked = {pos for pos, _ in obs.bombs} | {other.pos for other in obs.others}
    width, height = obs.field.shape
    actions: list[str] = []
    for action in MOVES:
        x, y = step(obs.me.pos, action)
        if (
            0 <= x < width
            and 0 <= y < height
            and obs.field[x, y] == FREE
            and (x, y) not in blocked
        ):
            actions.append(action)
    actions.append(WAIT)
    if obs.me.can_bomb:
        actions.append(BOMB)
    return actions


def crate_cells(field: NDArray[np.int64]) -> frozenset[Pos]:
    return frozenset((int(x), int(y)) for x, y in np.argwhere(field == CRATE))


@dataclass(frozen=True)
class PendingBomb:
    """A bomb, real or hypothetical, that detonates after action ``explodes``."""

    pos: Pos
    explodes: int


def own_bomb(pos: Pos) -> PendingBomb:
    """The bomb the current action would drop at ``pos``."""
    return PendingBomb(pos, BOMB_TIMER)


class Timeline:
    """Which cells kill, and which cells block, at each upcoming offset.

    ``lethal[t]`` is the set of cells an agent must not stand on after action
    ``t``. ``bomb_until[c] = k`` blocks entering ``c`` for actions at offsets
    ``<= k``. ``crate_open[c] = t`` makes crate ``c`` enterable from offset
    ``t``. The horizon is the first offset after which nothing known is lethal.
    """

    def __init__(self) -> None:
        self.lethal: list[set[Pos]] = []
        self.bomb_until: dict[Pos, int] = {}
        self.crate_open: dict[Pos, int] = {}

    @property
    def horizon(self) -> int:
        return len(self.lethal)

    def is_lethal(self, pos: Pos, offset: int) -> bool:
        return offset < len(self.lethal) and pos in self.lethal[offset]

    def lethal_offsets(self, pos: Pos) -> list[int]:
        return [t for t, cells in enumerate(self.lethal) if pos in cells]

    def mark(self, cells: Iterable[Pos], first: int, last: int) -> None:
        """Make ``cells`` lethal at offsets ``first .. last`` inclusive."""
        while len(self.lethal) <= last:
            self.lethal.append(set())
        cells = tuple(cells)
        for offset in range(first, last + 1):
            self.lethal[offset].update(cells)

    def add_bomb(
        self, geometry: Geometry, crates: Container[Pos], bomb: PendingBomb
    ) -> None:
        blast = geometry.blast(bomb.pos)
        self.mark(blast, bomb.explodes, bomb.explodes + EXPLOSION_TIMER - 1)
        self.bomb_until[bomb.pos] = max(
            self.bomb_until.get(bomb.pos, -1), bomb.explodes
        )
        opens = bomb.explodes + 1
        for cell in blast:
            if cell in crates and self.crate_open.get(cell, opens + 1) > opens:
                self.crate_open[cell] = opens

    def copy(self) -> "Timeline":
        clone = Timeline()
        clone.lethal = [set(cells) for cells in self.lethal]
        clone.bomb_until = dict(self.bomb_until)
        clone.crate_open = dict(self.crate_open)
        return clone

    def with_bombs(
        self, geometry: Geometry, crates: Container[Pos], bombs: Iterable[PendingBomb]
    ) -> "Timeline":
        """A copy with hypothetical bombs added; ``self`` is left unchanged."""
        clone = self.copy()
        for bomb in bombs:
            clone.add_bomb(geometry, crates, bomb)
        return clone


def danger_timeline(
    geometry: Geometry,
    crates: Container[Pos],
    bombs: Iterable[tuple[Pos, int]],
    explosion_map: NDArray[np.float64],
) -> Timeline:
    """Forecast every known hazard: live explosions and ticking bombs."""
    timeline = Timeline()
    for x, y in np.argwhere(explosion_map >= 1):
        value = int(explosion_map[x, y])
        timeline.mark([(int(x), int(y))], 0, value - 1)
    for pos, timer in bombs:
        timeline.add_bomb(geometry, crates, PendingBomb(pos, timer))
    return timeline
