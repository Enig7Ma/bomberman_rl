"""bfs_agent: a search-based rule agent used as a benchmark and training opponent.

Not a course submission -- the course requires a learned model. See
dev/rule_based_agent.md Part 2 for the plan this implements.

Each step: forecast every known hazard (``world_model``), grade each legal
action by whether an escape survives it (``safety``), then among the safest
actions pick the one that brings the most valuable target closest -- a coin,
a cell from which a bomb would destroy crates (``planning``), or one from
which it would catch an opponent (``attack``). A bomb is additionally worth
whatever it does to the opponents right now.

The framework imports this module as ``agent_code.bfs_agent.callbacks`` and
passes a ``SimpleNamespace`` as ``self``; ``AgentSelf`` declares the attributes
this agent reads and writes on it.
"""

import logging
import random
from typing import Any, Protocol

from .attack import bomb_attack_value, hunt_targets
from .params import Params
from .planning import (
    CoinRoute,
    DistanceCache,
    action_values,
    bomb_spots,
    choose,
    coin_targets,
)
from .safety import Board, assess_actions, safest, threat_bombs
from .world_model import BOMB, Geometry, Observation, danger_timeline


class AgentSelf(Protocol):
    logger: logging.Logger
    train: bool
    params: Params
    rng: random.Random
    geometry: Geometry | None
    distances: DistanceCache
    route: CoinRoute
    round: int


def setup(self: AgentSelf) -> None:
    self.params = Params.from_env()
    # A private generator: never touch the global ``random``/NumPy state that
    # other agents in the same process share.
    self.rng = random.Random(self.params.seed)
    self.geometry = None
    self.distances = DistanceCache()
    _start_round(self, 0)


def _start_round(self: AgentSelf, number: int) -> None:
    # The stock framework calls ``setup`` once and reuses the agent for every
    # round, so per-round state must be reset here.
    self.round = number
    self.route = CoinRoute()


def _geometry(self: AgentSelf, obs: Observation) -> Geometry:
    if self.geometry is None or not self.geometry.matches(obs.field):
        self.geometry = Geometry(obs.field)
    return self.geometry


def act(self: AgentSelf, game_state: dict[str, Any]) -> str:
    obs = Observation.from_game_state(game_state)
    if obs.round != self.round:
        _start_round(self, obs.round)
    params = self.params
    geometry = _geometry(self, obs)
    board = Board(obs, geometry)
    timeline = danger_timeline(geometry, board.crates, obs.bombs, obs.explosion_map)
    safe = safest(assess_actions(obs, board, timeline, threat_bombs(obs)))
    if safe[0].tier == 0:
        self.logger.info(
            f"step {obs.step}: no known escape, playing for time with"
            f" {[a.action for a in safe]}"
        )
    self.distances.sync(board)
    targets = coin_targets(obs, board, self.distances, params, self.route)
    targets += bomb_spots(obs, board, self.distances, timeline, params)
    targets += hunt_targets(obs, board, self.distances, params)
    actions = [a.action for a in safe]
    values = action_values(obs, actions, targets, self.distances, params)
    if BOMB in values:
        values[BOMB] += bomb_attack_value(obs, board, timeline, params)
        # A bomb with nothing to gain only costs us the bomb and a blocked tile.
        if len(values) > 1 and values[BOMB] <= 0.0:
            del values[BOMB]
    return choose(values, self.rng)
