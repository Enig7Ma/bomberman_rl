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

from .world_model import BOMB, WAIT, Observation, legal_actions, step


class AgentSelf(Protocol):
    logger: logging.Logger
    train: bool
    rng: random.Random


def setup(self: AgentSelf) -> None:
    # A private generator: never touch the global ``random``/NumPy state that
    # other agents in the same process share.
    self.rng = random.Random()


def act(self: AgentSelf, game_state: dict[str, Any]) -> str:
    obs = Observation.from_game_state(game_state)
    candidates = [
        action
        for action in legal_actions(obs)
        if action != BOMB and obs.explosion_map[step(obs.me.pos, action)] < 1
    ]
    return self.rng.choice(candidates) if candidates else WAIT
