"""Hand-built boards and game states for the ``bfs_agent`` tests.

Boards are drawn as text, one string per *row* (``y``), so they read the same
way the GUI shows them. ``field[x, y]`` indexing is handled here.

    #  stone wall      .  free      c  crate
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from agent_code.bfs_agent.world_model import CRATE, FREE, WALL, Pos

_CELLS = {"#": WALL, ".": FREE, "c": CRATE}


def parse_board(rows: Sequence[str]) -> NDArray[np.int64]:
    """Turn rows of ``#``/``.``/``c`` into a ``field`` array indexed ``[x, y]``."""
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("board rows must all have the same width")
    field = np.zeros((width, len(rows)), dtype=np.int64)
    for y, row in enumerate(rows):
        for x, cell in enumerate(row):
            field[x, y] = _CELLS[cell]
    return field


def arena(crates: Sequence[Pos] = ()) -> NDArray[np.int64]:
    """The real 17x17 stone layout, with optional crates."""
    field = np.zeros((17, 17), dtype=np.int64)
    field[0, :] = WALL
    field[-1, :] = WALL
    field[:, 0] = WALL
    field[:, -1] = WALL
    for x in range(17):
        for y in range(17):
            if (x + 1) * (y + 1) % 2 == 1:
                field[x, y] = WALL
    for pos in crates:
        field[pos] = CRATE
    return field


def game_state(
    field: NDArray[np.int64],
    me: Pos,
    *,
    can_bomb: bool = True,
    others: Sequence[Pos] = (),
    others_can_bomb: bool = True,
    bombs: Sequence[tuple[Pos, int]] = (),
    coins: Sequence[Pos] = (),
    explosions: Sequence[tuple[Pos, int]] = (),
    step: int = 1,
    round_number: int = 1,
) -> dict[str, Any]:
    """A ``game_state`` dict shaped exactly like ``get_state_for_agent``'s."""
    explosion_map = np.zeros(field.shape)
    for pos, value in explosions:
        explosion_map[pos] = value
    return {
        "round": round_number,
        "step": step,
        "field": np.array(field),
        "self": ("me", 0, can_bomb, me),
        "others": [
            (f"other_{i}", 0, others_can_bomb, pos) for i, pos in enumerate(others)
        ],
        "bombs": list(bombs),
        "coins": list(coins),
        "user_input": None,
        "explosion_map": explosion_map,
    }
