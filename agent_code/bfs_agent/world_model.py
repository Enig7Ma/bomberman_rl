"""Typed view of the observation, and the geometry the rest of the agent builds on.

The framework hands ``act`` an untyped dict whose coordinates are a mixture of
Python and NumPy integers. ``Observation.from_game_state`` is the single place
that dict is read; everything downstream works on plain tuples of ``int``.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray


Pos = tuple[int, int]

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
