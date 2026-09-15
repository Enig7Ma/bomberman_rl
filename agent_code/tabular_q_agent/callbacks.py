"""tabular_q_agent: tabular Q-learning over a compact abstract state.

Implements ``dev/tabiular_q-learning.md``. The agent does not learn yet: each
step ``features.Extractor`` grades every legal action with the safety search
copied from ``bfs_agent`` (``core/``), derives the action mask (``mask``) and
the categorical state features, and the agent picks an allowed action
uniformly at random with a private RNG. The features are computed but unused
until the Q-table arrives (plan step Q3). The random policy stays available as
the *safe-random* control, which any trained table must beat.

The framework imports this module as ``agent_code.tabular_q_agent.callbacks``
and calls each function with a ``types.SimpleNamespace`` as ``self``. It only
checks the *number* of parameters (``agents.AGENT_API``), so the annotations
below are free to be precise. ``AgentSelf`` declares what lives on that
namespace.

Coordinates in ``GameState`` are declared as ``int``, but the engine hands out
a mixture of Python and NumPy integers; ``core.world_model.Observation``
converts them once.
"""

import logging
import random
from typing import Final, Literal, Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray

from .config import Config
from .core.world_model import Observation
from .features import ENCODINGS, Extractor

Action = Literal["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
ACTIONS: Final[tuple[Action, ...]] = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
_BY_NAME: Final[dict[str, Action]] = {action: action for action in ACTIONS}

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
    """Attributes on the ``self`` namespace the framework passes in."""

    logger: logging.Logger
    """Preset: writes to ``agent_code/tabular_q_agent/logs/<agent_name>.log``."""
    train: bool
    """Preset: True when started with ``--train`` covering this agent."""
    config: Config
    rng: random.Random
    """Private generator: never touch the global ``random``/NumPy state that
    other agents in the same process share."""
    extractor: Extractor
    """Features and the action mask; keeps geometry and distance caches."""


def setup(self: AgentSelf) -> None:
    """Called once, before the first round, in both play and training mode."""
    self.config = Config.from_env()
    self.rng = random.Random(self.config.seed)
    self.extractor = Extractor(
        ENCODINGS[self.config.encoding], self.config.mask, self.rng
    )


def act(self: AgentSelf, game_state: GameState) -> Action:
    """Called once per step; return the action to take.

    Must return within ``settings.TIMEOUT`` (0.5 s) outside training mode, or
    the engine executes ``WAIT`` instead and shortens the next step's budget
    by the overrun. There is no limit in training mode.
    """
    obs = Observation.from_game_state(game_state)
    extracted = self.extractor.extract(obs)
    if extracted.best_tier == 0:
        self.logger.info(
            f"step {obs.step}: no known escape, playing for time with"
            f" {list(extracted.allowed)}"
        )
    return _BY_NAME[self.rng.choice(extracted.allowed)]
