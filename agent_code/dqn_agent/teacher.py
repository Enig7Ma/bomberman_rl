"""The search agent's own policy, as an exploration behaviour for training.

``core/`` is `bfs_agent`'s search code, vendored whole, so its decision rule can
be reproduced here exactly: grade every legal action for safety, then among the
safest pick the one that brings the most valuable target -- a coin, a cell to
bomb crates from, a cell to catch an opponent from -- closest, with a bomb also
worth what it does to the opponents right now.

Why a learner should want it (dev/dqn.md D10, "teacher transitions"): Q-learning
is off-policy, so the values it learns are those of *its own* greedy policy no
matter who collected the data. A uniform-over-the-mask explorer almost never
reaches the states a competent agent spends its time in -- standing next to a
crate wall with an escape booked, or cornering an opponent -- so those states
carry almost no training signal. Playing a share of the steps with this policy
moves the data there, and the learner is free to disagree with it afterwards;
it is a *behaviour* policy, never a target and never a prior on the weights.

This module is imported only by ``train.py``. Inference never constructs a
Teacher: ``act`` outside training reads the network and nothing else. Whenever
a run uses it, the share is recorded in the run's config and reported.
"""

import random

from .core.attack import bomb_attack_value, hunt_targets
from .core.params import Params
from .core.planning import (
    CoinRoute,
    DistanceCache,
    action_values,
    bomb_spots,
    choose,
    coin_targets,
)
from .core.safety import Board, assess_actions, safest, threat_bombs
from .core.world_model import BOMB, Geometry, Observation, danger_timeline


class Teacher:
    """One agent's worth of search state: geometry, distances and coin route.

    Kept across steps exactly as `bfs_agent` keeps it, and reset per round,
    because the coin route is a commitment that only makes sense within a round.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.params = Params()
        self.rng = random.Random(seed)
        self.geometry: Geometry | None = None
        self.distances = DistanceCache()
        self.route = CoinRoute()
        self.round = 0

    def _geometry_for(self, obs: Observation) -> Geometry:
        if self.geometry is None or not self.geometry.matches(obs.field):
            self.geometry = Geometry(obs.field)
        return self.geometry

    def act(self, obs: Observation) -> str:
        if obs.round != self.round:
            self.round = obs.round
            self.route = CoinRoute()
        params = self.params
        geometry = self._geometry_for(obs)
        board = Board(obs, geometry)
        timeline = danger_timeline(geometry, board.crates, obs.bombs, obs.explosion_map)
        safe = safest(assess_actions(obs, board, timeline, threat_bombs(obs)))
        self.distances.sync(board)
        targets = coin_targets(obs, board, self.distances, params, self.route)
        targets += bomb_spots(obs, board, self.distances, timeline, params)
        targets += hunt_targets(obs, board, self.distances, params)
        actions = [a.action for a in safe]
        values = action_values(obs, actions, targets, self.distances, params)
        if BOMB in values:
            values[BOMB] += bomb_attack_value(obs, board, timeline, params)
            if len(values) > 1 and values[BOMB] <= 0.0:
                del values[BOMB]
        return choose(values, self.rng)
