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
in ``TABULAR_Q_AGENT_MODEL``. In training mode (``train.py``) the agent also
explores with probability ``epsilon`` and hands every step to a
``transitions.Trainer``, which turns the framework's callbacks into Q-updates.
Outside training it never explores and never writes. ``policy="random"``
keeps the safe-random control of step Q0 and ``policy="heuristic"`` the
hand-ordered control of §5.9: both ignore the table for acting (it still
learns off-policy when training). ``teacher_share`` mixes heuristic actions
into training only, as the plan's teacher-guided exploration (Q11).

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
from pathlib import Path
from typing import Final, Literal, Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray

from .config import MODEL_ENV_VAR, Config, model_path
from .core.world_model import Observation
from .features import ENCODINGS, Encoding, Extractor
from .heuristic import heuristic_action
from .learner import Learner, Selection
from .qtable import ACTION_COLUMN, QTable
from .symmetry import IDENTITY, canonical, from_canonical, to_canonical
from .transitions import Trainer

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
    model_file: Path
    table: QTable
    learner: Learner
    round: int
    """The round ``act`` last saw; the framework reuses one agent for all."""
    trainer: Trainer | None
    """Set by ``train.setup_training``; None outside training."""


def setup(self: AgentSelf) -> None:
    """Called once, before the first round, in both play and training mode."""
    self.config = Config.from_env()
    self.rng = random.Random(self.config.seed)
    self.encoding = ENCODINGS[self.config.encoding]
    self.extractor = Extractor(self.encoding, self.config.mask, self.rng)
    self.model_file, explicit = model_path()
    self.table = _load_table(self, explicit)
    self.learner = Learner(
        self.table,
        gamma=self.config.gamma,
        alpha_omega=self.config.alpha_omega,
        alpha_min=self.config.alpha_min,
        rng=self.rng,
    )
    self.round = 0
    self.trainer = None


def _load_table(self: AgentSelf, explicit: bool) -> QTable:
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
    path = self.model_file
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
    trainer = self.trainer
    if obs.round != self.round:
        self.round = obs.round
        if trainer is not None:
            trainer.begin_round(obs.round, [other.name for other in obs.others])

    extracted = self.extractor.extract(obs)
    if extracted.best_tier == 0:
        self.logger.info(
            f"step {obs.step}: no known escape, playing for time with"
            f" {list(extracted.allowed)}"
        )
    if self.config.symmetry:
        state, symmetry = canonical(extracted.features, self.encoding)
    else:
        state, symmetry = self.encoding.encode(extracted.features), IDENTITY
    allowed = [ACTION_COLUMN[to_canonical(a, symmetry)] for a in extracted.allowed]
    if trainer is not None:
        trainer.observe(
            obs.round,
            obs.step,
            state,
            allowed,
            extracted.coin_distance,
            bomb_hits=extracted.bomb_hits,
            crate_distance=extracted.crate_distance,
        )

    teaching = (
        trainer is not None
        and self.config.teacher_share > 0.0
        and self.rng.random() < self.config.teacher_share
    )
    if self.config.policy == "random":
        selection = Selection(self.rng.choice(allowed), explored=False, unseen=False)
    elif self.config.policy == "heuristic" or teaching:
        choice = heuristic_action(extracted.features, extracted.allowed, self.rng)
        column = ACTION_COLUMN[to_canonical(choice, symmetry)]
        # A teacher step is a deviation from the table's own policy, so the
        # metrics count it as exploration, like an epsilon step.
        selection = Selection(column, explored=teaching, unseen=False)
    else:
        epsilon = self.config.epsilon if trainer is not None else 0.0
        selection = self.learner.select(state, allowed, epsilon)
    played = _BY_NAME[from_canonical(ACTIONS[selection.action], symmetry)]

    if trainer is not None:
        trainer.chose(selection, played)
    return played
