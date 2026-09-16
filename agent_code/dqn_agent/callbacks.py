"""D0 safe-random control over the shared E3 safety mask.

There is no network or training yet. Both policy settings use uniform allowed
actions; requesting learned play logs that this is still the scaffold. Only
this agent's vendored modules, NumPy and the framework are needed at runtime.
"""

import logging
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from .config import Config, model_path
from .core.world_model import Observation
from .features import ENCODINGS, Encoding, Extractor

Action = Literal["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]


class AgentSelf(Protocol):
    logger: logging.Logger
    train: bool
    config: Config
    rng: random.Random
    encoding: Encoding
    extractor: Extractor
    model_file: Path
    round: int


def setup(self: AgentSelf) -> None:
    self.config = Config.from_env()
    self.rng = random.Random(self.config.seed)
    self.encoding = ENCODINGS[self.config.encoding]
    self.extractor = Extractor(self.encoding, self.config.mask, self.rng)
    self.model_file, _ = model_path()
    self.round = 0
    if self.config.policy == "learned":
        self.logger.warning("D0 scaffold has no Q-network; playing safe-random")


def act(self: AgentSelf, game_state: Mapping[str, Any]) -> Action:
    obs = Observation.from_game_state(game_state)
    if obs.round != self.round:
        self.round = obs.round
        self.extractor = Extractor(self.encoding, self.config.mask, self.rng)
    extracted = self.extractor.extract(obs)
    return cast(Action, self.rng.choice(extracted.allowed))
