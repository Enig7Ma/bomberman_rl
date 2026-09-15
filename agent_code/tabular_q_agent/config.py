"""Hyperparameters, with an environment override for experiments.

The tournament harness can only name an agent directory, so a sweep sets
``TABULAR_Q_AGENT_PARAMS`` to a JSON object of field overrides before launching
it; worker processes inherit the environment. When the variable is unset --
always the case under the official framework -- the frozen defaults apply.

Fields are added in the plan step that first uses them
(``dev/tabiular_q-learning.md``); unknown keys are rejected so a typo in a
sweep fails loudly instead of silently running the defaults.
"""

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Literal, cast, get_args

ENV_VAR = "TABULAR_Q_AGENT_PARAMS"

# The Q-table to load (and, in training, to save). Unset under the official
# framework, where the table shipped inside the agent directory is used.
MODEL_ENV_VAR = "TABULAR_Q_AGENT_MODEL"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "model" / "q_table.npz"

# How ``act`` chooses: from the Q-table, or uniformly over the mask (the
# safe-random control of plan step Q0, which ignores the table).
Policy = Literal["learned", "random"]
POLICIES: tuple[Policy, ...] = get_args(Policy)

# Which safety tiers the agent may choose from (plan §5.2): the best tier
# available, anything that survives static opponents (tier >= 2), anything with
# some known escape (tier >= 1), or every legal action.
MaskVariant = Literal["best_tier", "min_tier_2", "any_escape", "legal"]
MASK_VARIANTS: tuple[MaskVariant, ...] = get_args(MaskVariant)

# Which state abstraction to use (plan §5.3); ``features.ENCODINGS`` defines
# them. E3 is the submission's; E1 and E2 are small enough to debug by hand.
EncodingName = Literal["E1", "E2", "E3"]
ENCODING_NAMES: tuple[EncodingName, ...] = get_args(EncodingName)


def model_path(environ: Mapping[str, str] | None = None) -> tuple[Path, bool]:
    """The Q-table's path, and whether ``MODEL_ENV_VAR`` named it explicitly."""
    raw = (os.environ if environ is None else environ).get(MODEL_ENV_VAR, "").strip()
    if raw:
        return Path(raw), True
    return DEFAULT_MODEL_PATH, False


# Overrides come from JSON, so the annotations alone guarantee nothing.
def _is_seed(value: object) -> bool:
    return value is None or (isinstance(value, int) and not isinstance(value, bool))


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


@dataclass(frozen=True)
class Config:
    mask: MaskVariant = "best_tier"
    encoding: EncodingName = "E3"
    policy: Policy = "learned"
    # Discount of the Q-learning target (plan §5.6).
    gamma: float = 0.99
    # Step size ``max(alpha_min, (1 + visits) ** -alpha_omega)``: polynomial
    # decay per state-action, with a floor so values keep tracking opponents
    # that change between curriculum stages.
    alpha_omega: float = 0.7
    alpha_min: float = 0.05
    # Seed for the agent's private RNG; None draws one from the operating system.
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.mask not in MASK_VARIANTS:
            raise ValueError(f"mask must be one of {MASK_VARIANTS}, got {self.mask!r}")
        if self.encoding not in ENCODING_NAMES:
            raise ValueError(
                f"encoding must be one of {ENCODING_NAMES}, got {self.encoding!r}"
            )
        if self.policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, got {self.policy!r}")
        for name, value, low, high in (
            ("gamma", self.gamma, 0.0, 1.0),
            ("alpha_omega", self.alpha_omega, 0.0, 1.0),
            ("alpha_min", self.alpha_min, 0.0, 1.0),
        ):
            if not _is_number(value) or not low < value <= high:
                raise ValueError(f"{name} must be a number in ({low}, {high}]")
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
