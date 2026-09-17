"""DQN inference settings and repository-relative file locations.

The learned policy uses NumPy weights; random is an explicit control policy.
Paths do not depend on the framework's cwd.
"""

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Literal, cast, get_args

from .encoder import ENCODERS
from .mask import MASK_VARIANTS, MaskVariant

AGENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AGENT_DIR.parents[1]
ENV_PREFIX = AGENT_DIR.name.upper()
ENV_VAR = f"{ENV_PREFIX}_PARAMS"
MODEL_ENV_VAR = f"{ENV_PREFIX}_MODEL"
METRICS_ENV_VAR = f"{ENV_PREFIX}_METRICS"
DEFAULT_MODEL_PATH = AGENT_DIR / "model" / "q_net.npz"
DEFAULT_METRICS_PATH = AGENT_DIR / "logs" / "train_metrics.jsonl"

Policy = Literal["learned", "random"]
POLICIES: tuple[Policy, ...] = get_args(Policy)


def _env_path(
    variable: str, default: Path, environ: Mapping[str, str] | None
) -> tuple[Path, bool]:
    raw = (os.environ if environ is None else environ).get(variable, "").strip()
    if not raw:
        return default, False
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path), True


def model_path(environ: Mapping[str, str] | None = None) -> tuple[Path, bool]:
    """Model path and whether it was explicitly configured."""
    return _env_path(MODEL_ENV_VAR, DEFAULT_MODEL_PATH, environ)


def metrics_path(environ: Mapping[str, str] | None = None) -> Path:
    return _env_path(METRICS_ENV_VAR, DEFAULT_METRICS_PATH, environ)[0]


def _is_seed(value: object) -> bool:
    return value is None or (isinstance(value, int) and not isinstance(value, bool))


def _is_encoder(value: object) -> bool:
    return isinstance(value, str) and value in ENCODERS


def _is_init_seed(value: object) -> bool:
    return value is None or (type(value) is int and value >= 0)


@dataclass(frozen=True)
class Config:
    mask: MaskVariant = "best_tier"
    encoding: Literal["E3"] = "E3"
    encoder: str = "onehot_e3"
    policy: Policy = "learned"
    seed: int | None = None
    init_seed: int | None = None
    gamma: float = 0.99
    lr: float = 3e-4
    batch_size: int = 64
    replay_size: int = 100_000
    warmup: int = 5_000
    train_every: int = 4
    target_every: int = 1_000
    grad_clip: float = 10.0
    c_coin: float = 0.5
    crate_aid: float = 0.0
    death_aid: float = 0.0
    epsilon_start: float = 0.3
    epsilon_end: float = 0.05
    epsilon_fraction: float = 0.6
    stage: int = 0
    stage_transitions: int = 50_000
    save_every: int = 1
    replay_save_every: int = 50
    probe_every: int = 10_000
    probe_path: str | None = None

    def __post_init__(self) -> None:
        if self.mask not in MASK_VARIANTS:
            raise ValueError(f"mask must be one of {MASK_VARIANTS}, got {self.mask!r}")
        if self.encoding != "E3":
            raise ValueError(f"encoding must be E3, got {self.encoding!r}")
        if not _is_encoder(self.encoder):
            raise ValueError(
                f"encoder must be one of {tuple(ENCODERS)}, got {self.encoder!r}"
            )
        if self.policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, got {self.policy!r}")
        if not _is_seed(self.seed):
            raise ValueError(f"seed must be an integer or null, got {self.seed!r}")
        if not _is_init_seed(self.init_seed):
            raise ValueError("init_seed must be a nonnegative integer or null")

        for name in (
            "batch_size",
            "replay_size",
            "train_every",
            "target_every",
            "warmup",
            "stage_transitions",
            "save_every",
            "replay_save_every",
            "probe_every",
        ):
            value = getattr(self, name)
            minimum = 0 if name == "warmup" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in (
            "gamma",
            "lr",
            "grad_clip",
            "c_coin",
            "crate_aid",
            "death_aid",
            "epsilon_start",
            "epsilon_end",
            "epsilon_fraction",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if not 0 <= self.gamma <= 1 or self.lr <= 0 or self.grad_clip <= 0:
            raise ValueError(
                "gamma must be in [0,1]; lr and grad_clip must be positive"
            )
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("expected 0 <= epsilon_end <= epsilon_start <= 1")
        if type(self.stage) is not int or not 0 <= self.stage <= 255:
            raise ValueError("stage must be an integer in 0..255")
        if self.probe_path is not None and type(self.probe_path) is not str:
            raise ValueError("probe_path must be a path string or null")
        if not 0 < self.epsilon_fraction <= 1:
            raise ValueError("epsilon_fraction must be in (0,1]")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Config":
        raw = (os.environ if environ is None else environ).get(ENV_VAR, "")
        if not raw.strip():
            return cls()
        parsed: object = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError(f"{ENV_VAR} must be a JSON object, got {raw!r}")
        overrides = cast(dict[str, object], parsed)
        unknown = set(overrides) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"{ENV_VAR} names unknown parameters {sorted(unknown)}")
        return replace(cls(), **overrides)
