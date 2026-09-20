"""Play callbacks: canonical E3 inputs and masked NumPy Q-network inference.

Missing/broken default weights fall back to safe-random with an error log.
Explicit model paths are strict; policy="random" remains the control.
"""

import logging
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from .config import Config, model_path
from .core.world_model import ACTIONS, Observation
from .encoder import ENCODERS, Encoder
from .features import ENCODINGS, Encoding, Extracted, Extractor
from .network import QFunction, QNetwork, masked_greedy
from .symmetry import Symmetry, canonical, from_canonical, to_canonical

Action = Literal["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]


class TrainingPolicy(Protocol):
    def select(
        self, obs: Observation, extracted: Extracted, index: int, symmetry: Symmetry
    ) -> str: ...


class AgentSelf(Protocol):
    logger: logging.Logger
    train: bool
    config: Config
    rng: random.Random
    encoding: Encoding
    extractor: Extractor
    model_file: Path
    round: int
    encoder: Encoder
    q_function: QFunction | None
    trainer: TrainingPolicy | None


def setup(self: AgentSelf) -> None:
    self.config = Config.from_env()
    self.rng = random.Random(self.config.seed)
    self.encoding = ENCODINGS[self.config.encoding]
    self.extractor = Extractor(self.encoding, self.config.mask, self.rng)
    self.encoder = ENCODERS[self.config.encoder]
    self.model_file, explicit = model_path()
    self.q_function = _load_network(self, explicit)
    self.round = 0
    self.trainer = None


def _load_network(self: AgentSelf, explicit: bool) -> QNetwork | None:
    if self.train and (self.model_file.parent / "checkpoint.pt").exists():
        return None
    if not self.model_file.exists():
        if explicit and not self.train:
            raise FileNotFoundError(self.model_file)
        if self.train:
            return QNetwork.random(self.encoder, self.config.init_seed)
        self.logger.error(f"no Q-network at {self.model_file}; playing safe-random")
        return None
    try:
        network = QNetwork.load(self.model_file, self.encoder)
    except Exception as error:
        if explicit or self.train:
            raise
        self.logger.error(
            f"cannot use Q-network at {self.model_file} ({error}); playing safe-random"
        )
        return None
    self.logger.info(
        f"loaded Q-network {self.model_file}: schema {self.encoder.schema_id}"
    )
    return network


def act(self: AgentSelf, game_state: Mapping[str, Any]) -> Action:
    obs = Observation.from_game_state(game_state)
    if obs.round != self.round:
        self.round = obs.round
        self.extractor = Extractor(self.encoding, self.config.mask, self.rng)
    extracted = self.extractor.extract(obs)
    if self.train and self.trainer is not None:
        index, symmetry = canonical(extracted.features, self.encoding)
        return cast(Action, self.trainer.select(obs, extracted, index, symmetry))
    if self.config.policy == "random" or self.q_function is None:
        return cast(Action, self.rng.choice(extracted.allowed))
    index, symmetry = canonical(extracted.features, self.encoding)
    x = self.encoder.encode(self.encoding.decode(index), extracted)
    allowed = [ACTIONS.index(to_canonical(a, symmetry)) for a in extracted.allowed]
    action = masked_greedy(self.q_function.values(x), allowed, self.rng)
    return cast(Action, from_canonical(ACTIONS[action], symmetry))
