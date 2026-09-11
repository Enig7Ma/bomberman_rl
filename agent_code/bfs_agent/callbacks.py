"""bfs_agent: a search-based rule agent used as a benchmark and training opponent.

Not a course submission -- the course requires a learned model. See
dev/rule_based_agent.md Part 2 for the plan this implements.

The framework imports this module as ``agent_code.bfs_agent.callbacks`` and
passes a ``SimpleNamespace`` as ``self``; ``AgentSelf`` declares the attributes
this agent reads and writes on it.
"""

import logging
import random
from typing import Any, Protocol

from .safety import Board, assess_actions, safest, threat_bombs
from .world_model import Geometry, Observation, danger_timeline


class AgentSelf(Protocol):
    logger: logging.Logger
    train: bool
    rng: random.Random
    geometry: Geometry | None


def setup(self: AgentSelf) -> None:
    # A private generator: never touch the global ``random``/NumPy state that
    # other agents in the same process share.
    self.rng = random.Random()
    self.geometry = None


def _geometry(self: AgentSelf, obs: Observation) -> Geometry:
    if self.geometry is None or not self.geometry.matches(obs.field):
        self.geometry = Geometry(obs.field)
    return self.geometry


def act(self: AgentSelf, game_state: dict[str, Any]) -> str:
    obs = Observation.from_game_state(game_state)
    geometry = _geometry(self, obs)
    board = Board(obs, geometry)
    timeline = danger_timeline(geometry, board.crates, obs.bombs, obs.explosion_map)
    choices = safest(assess_actions(obs, board, timeline, threat_bombs(obs)))
    if choices[0].tier == 0:
        self.logger.info(
            f"step {obs.step}: no known escape, playing for time with"
            f" {[a.action for a in choices]}"
        )
    return self.rng.choice(choices).action
