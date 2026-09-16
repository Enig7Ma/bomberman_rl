"""DQN scaffold settings and repository-relative file locations (plan D0).

The scaffold plays safe-random. Network and training parameters are added only
when their implementations exist. Paths do not depend on the framework's cwd.
"""

import json
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


@dataclass(frozen=True)
class Config:
    mask: MaskVariant = "best_tier"
    encoding: Literal["E3"] = "E3"
    encoder: str = "onehot_e3"
    policy: Policy = "random"
    seed: int | None = None

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
