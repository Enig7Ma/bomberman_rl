"""Agent callbacks that are always loaded: ``setup`` and ``act``.

This agent is the keyboard-controlled player (``--agents user_agent`` in the
GUI): ``act`` returns whatever key was pressed. Everything else in this file
and in ``train.py`` is a typed skeleton meant to be copied into a new
``agent_code/<name>/`` directory as a starting point.

The framework imports this module as ``agent_code.<name>.callbacks`` and calls
each function with a ``types.SimpleNamespace`` as ``self``. It only checks the
*number* of parameters (``agents.AGENT_API``), so the annotations below are
free to be precise. ``AgentSelf`` declares what lives on that namespace; add
every attribute your agent stores (``self.model = ...``) to it, so pyright
checks the attribute accesses.

Coordinates in ``GameState`` are declared as ``int``, but the engine hands out
a mixture of Python and NumPy integers. Convert them once with ``int(...)`` if
that matters (e.g. for hashing into dicts).
"""

import logging
from typing import Final, Literal, Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray

Action = Literal["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
ACTIONS: Final[tuple[Action, ...]] = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")

Pos = tuple[int, int]
# (name, score, bomb available, (x, y))
PlayerState = tuple[str, int, bool, Pos]


class GameState(TypedDict):
    """The observation built by ``environment.BombeRLeWorld.get_state_for_agent``."""

    round: int
    """Round number since the environment was launched, starting at 1."""
    step: int
    """Step within the round, starting at 1."""
    field: NDArray[np.int64]
    """Board indexed ``[x, y]``: ``1`` crate, ``-1`` stone wall, ``0`` free."""
    self: PlayerState
    """This agent. The bool is True while no own bomb is ticking."""
    others: list[PlayerState]
    """Opponents still alive, in the same format."""
    bombs: list[tuple[Pos, int]]
    """``((x, y), timer)`` per ticking bomb; timer 0 explodes after this step."""
    coins: list[Pos]
    """Collectable (revealed) coins only."""
    explosion_map: NDArray[np.float64]
    """Per cell: ``v >= 1`` is lethal after the coming action, ``0`` is harmless."""
    user_input: str | None
    """Key pressed in the GUI, mapped via ``settings.INPUT_MAP``; None if no key."""


class AgentSelf(Protocol):
    """Attributes on the ``self`` namespace the framework passes in.

    ``logger`` and ``train`` are preset by ``agents.AgentRunner``; declare your
    own attributes below them.
    """

    logger: logging.Logger
    """Writes to ``agent_code/<name>/logs/<agent_name>.log``."""
    train: bool
    """True when started with ``--train`` covering this agent."""


def setup(self: AgentSelf) -> None:
    """Called once, before the first round, in both play and training mode.

    Initialise everything ``act`` needs and store it on ``self``, e.g. load
    trained parameters when ``self.train`` is False. Files are best resolved
    relative to ``pathlib.Path(__file__).parent`` so they are found regardless
    of the working directory.

    The framework reuses this one ``self`` for every round; reset per-round
    state in ``act`` when ``game_state["round"]`` changes.
    """


def act(self: AgentSelf, game_state: GameState) -> Action:
    """Called once per step; return the action to take.

    Must return within ``settings.TIMEOUT`` (0.5 s) outside training mode, or
    the engine executes ``WAIT`` instead and shortens the next step's budget
    by the overrun. There is no limit in training mode.

    This agent plays the key pressed in the GUI, and waits if there is none.
    """
    key = game_state["user_input"]
    for action in ACTIONS:
        if key == action:
            return action
    return "WAIT"
