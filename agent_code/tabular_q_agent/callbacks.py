"""tabular_q_agent: tabular Q-learning over a compact abstract state.

Implements ``dev/tabiular_q-learning.md``. Each step:

1. ``features.Extractor`` grades every legal action with the safety search
   copied from ``bfs_agent`` (``core/``) and derives the action mask and the
   categorical state features;
2. ``symmetry.canonical`` maps the features to the table row of their orbit
   under the board's rotations and reflections;
3. ``learner.Learner`` picks the allowed action with the highest Q-value (ties
   at random), which is mapped back to the real board.

The table comes from ``model/q_table.npz`` in this directory, or from the path
in ``TABULAR_Q_AGENT_MODEL``. The agent does not learn yet -- the training
callbacks arrive in plan step Q4 -- so it only reads a table and never
explores. ``policy="random"`` keeps the safe-random control of step Q0:
uniform over the mask, table ignored. A trained table must beat it.

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

from .config import MODEL_ENV_VAR, Config, model_path
from .core.world_model import Observation
from .features import ENCODINGS, Encoding, Extractor
from .learner import Learner
from .qtable import ACTION_COLUMN, QTable
from .symmetry import canonical, from_canonical, to_canonical

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
    encoding: Encoding
    extractor: Extractor
    """Features and the action mask; keeps geometry and distance caches."""
    table: QTable
    learner: Learner


def setup(self: AgentSelf) -> None:
    """Called once, before the first round, in both play and training mode."""
    self.config = Config.from_env()
    self.rng = random.Random(self.config.seed)
    self.encoding = ENCODINGS[self.config.encoding]
    self.extractor = Extractor(self.encoding, self.config.mask, self.rng)
    self.table = _load_table(self)
    self.learner = Learner(
        self.table,
        gamma=self.config.gamma,
        alpha_omega=self.config.alpha_omega,
        alpha_min=self.config.alpha_min,
        rng=self.rng,
    )


def _load_table(self: AgentSelf) -> QTable:
    """The Q-table to play from.

    A path named in ``TABULAR_Q_AGENT_MODEL`` is taken at its word: outside
    training it must exist and match the encoding, since evaluating a missing
    or stale checkpoint would silently measure the safe-random control
    instead. Training may start from a table that does not exist yet.

    The default path is what the tournament uses. There a missing or unusable
    table is logged as an error and the agent plays from an empty table
    (which is safe-random) rather than crash; the packaging test is what
    guarantees the table ships.
    """
    path, explicit = model_path()
    if not path.exists():
        if explicit and not self.train:
            raise FileNotFoundError(
                f"{MODEL_ENV_VAR} names {path}, which does not exist"
            )
        report = self.logger.info if self.train else self.logger.error
        report(f"no Q-table at {path}; starting from an empty table")
        return QTable.zeros(self.encoding)
    try:
        table = QTable.load(path, self.encoding)
    except Exception as error:
        if explicit or self.train:
            raise
        self.logger.error(
            f"cannot use the Q-table at {path} ({error}); playing from an empty table"
        )
        return QTable.zeros(self.encoding)
    self.logger.info(f"loaded {path}: {table.visited_states} visited states")
    return table


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
    if self.config.policy == "random":
        return _BY_NAME[self.rng.choice(extracted.allowed)]

    state, symmetry = canonical(extracted.features, self.encoding)
    allowed = [ACTION_COLUMN[to_canonical(a, symmetry)] for a in extracted.allowed]
    selection = self.learner.select(state, allowed)
    return _BY_NAME[from_canonical(ACTIONS[selection.action], symmetry)]
