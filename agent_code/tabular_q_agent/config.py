"""Hyperparameters and file locations, with environment overrides for experiments.

The tournament harness can only name an agent directory, so a sweep sets
``TABULAR_Q_AGENT_PARAMS`` to a JSON object of field overrides before launching
it; worker processes inherit the environment. When the variable is unset --
always the case under the official framework -- the frozen defaults apply.

Unknown keys are rejected, so a typo in a sweep fails loudly instead of
silently running the defaults.

Relative paths in ``TABULAR_Q_AGENT_MODEL`` and ``TABULAR_Q_AGENT_METRICS`` are
taken from the repository root. The framework only runs from there, and it
changes the working directory into the agent directory before every callback,
so the process's working directory would be the wrong anchor.
"""

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Literal, cast, get_args

from .mask import MASK_VARIANTS, MaskVariant

AGENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AGENT_DIR.parents[1]

# Environment variables are named after the agent directory -- TABULAR_Q_AGENT_*
# here -- so a copy of this agent under another name (a frozen training
# opponent, see ``training/frozen.py``) reads its own and never the learner's.
ENV_PREFIX = AGENT_DIR.name.upper()
ENV_VAR = f"{ENV_PREFIX}_PARAMS"

# The Q-table to load (and, in training, to save). Unset under the official
# framework, where the table shipped inside the agent directory is used.
MODEL_ENV_VAR = f"{ENV_PREFIX}_MODEL"
DEFAULT_MODEL_PATH = AGENT_DIR / "model" / "q_table.npz"
# Where training appends one JSON record per round.
METRICS_ENV_VAR = f"{ENV_PREFIX}_METRICS"
DEFAULT_METRICS_PATH = AGENT_DIR / "logs" / "train_metrics.jsonl"

# Which state abstraction to use; ``features.ENCODINGS`` defines
# them. E3 is the submission's; E1 and E2 are small enough to debug by hand.
EncodingName = Literal["E1", "E2", "E3"]
ENCODING_NAMES: tuple[EncodingName, ...] = get_args(EncodingName)

# How ``act`` chooses: from the Q-table, uniformly over the mask (the
# safe-random control), or by the fixed priority of ``heuristic`` on the same
# features (the hand-tuned control). The two controls ignore the table for
# acting.
Policy = Literal["learned", "random", "heuristic"]
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
    """The Q-table's path, and whether ``MODEL_ENV_VAR`` named it explicitly."""
    return _env_path(MODEL_ENV_VAR, DEFAULT_MODEL_PATH, environ)


def metrics_path(environ: Mapping[str, str] | None = None) -> Path:
    return _env_path(METRICS_ENV_VAR, DEFAULT_METRICS_PATH, environ)[0]


# Overrides come from JSON, so the annotations alone guarantee nothing.
def _is_seed(value: object) -> bool:
    return value is None or (isinstance(value, int) and not isinstance(value, bool))


def _is_number(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _is_str(value: object) -> bool:
    return isinstance(value, str)


def _is_bool(value: object) -> bool:
    return isinstance(value, bool)


@dataclass(frozen=True)
class Config:
    mask: MaskVariant = "best_tier"
    encoding: EncodingName = "E3"
    policy: Policy = "learned"
    # Share one table row between board rotations and reflections.
    # False indexes the raw features, as an ablation.
    symmetry: bool = True
    # Discount of the Q-learning target and of the shaping term.
    gamma: float = 0.99
    # Step size ``max(alpha_min, (1 + visits) ** -alpha_omega)``: polynomial
    # decay per state-action, with a floor so values keep tracking opponents
    # that change between curriculum stages.
    alpha_omega: float = 0.7
    alpha_min: float = 0.05
    # Probability of a uniform allowed action while training; 0 outside
    # training regardless. Constant for one agent's lifetime: the training
    # driver lowers it between chunks of rounds.
    epsilon: float = 0.1
    # Potential shaping ``c / (1 + d)`` on the walking distance ``d`` to the
    # nearest visible coin; 0 turns it off.
    coin_potential: float = 0.5
    # Objective-changing training aids, off by default: a reward
    # per crate destroyed, and one once per death (negative for a penalty).
    crate_aid: float = 0.0
    death_aid: float = 0.0
    # Objective-changing aid paid on the BOMB action itself, per live crate the
    # dropped bomb will destroy (only when the engine confirms the drop):
    # without an immediate payoff for demolition the table cannot tell moving
    # towards a bombing spot from waiting; a bomb for nothing earns nothing.
    bomb_aid: float = 0.0
    # Potential shaping ``c / (1 + d)`` on the distance to the best bombing spot
    # added to the coin potential; 0 turns it off. Use with ``bomb_aid``.
    spot_potential: float = 0.0
    # Share of training steps played by ``heuristic`` instead of the table
    # (teacher-guided exploration). Q-learning is off-policy, so the
    # table still learns the values of its own greedy policy, from data the
    # better policy collected. 0 outside training regardless.
    teacher_share: float = 0.0
    # Save the table after every ``save_every``-th round (training only).
    save_every: int = 1
    # Free-form label copied into every training record, e.g. the curriculum stage.
    stage: str = ""
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
        for name, value in (
            ("gamma", self.gamma),
            ("alpha_omega", self.alpha_omega),
            ("alpha_min", self.alpha_min),
        ):
            if not _is_number(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be a number in (0, 1], got {value!r}")
        for name, value in (
            ("epsilon", self.epsilon),
            ("teacher_share", self.teacher_share),
        ):
            if not _is_number(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a number in [0, 1], got {value!r}")
        for name, value in (
            ("coin_potential", self.coin_potential),
            ("spot_potential", self.spot_potential),
        ):
            if not _is_number(value) or value < 0.0:
                raise ValueError(f"{name} must be a number >= 0, got {value!r}")
        for name, value in (
            ("crate_aid", self.crate_aid),
            ("death_aid", self.death_aid),
            ("bomb_aid", self.bomb_aid),
        ):
            if not _is_number(value):
                raise ValueError(f"{name} must be a finite number, got {value!r}")
        if not _is_positive_int(self.save_every):
            raise ValueError(
                f"save_every must be an integer >= 1, got {self.save_every!r}"
            )
        if not _is_str(self.stage):
            raise ValueError(f"stage must be a string, got {self.stage!r}")
        if not _is_bool(self.symmetry):
            raise ValueError(f"symmetry must be true or false, got {self.symmetry!r}")
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
