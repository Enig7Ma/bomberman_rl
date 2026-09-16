"""Tunable weights, with an environment override for experiments.

The tournament harness can only name an agent directory, so a parameter sweep
sets ``BFS_AGENT_PARAMS`` to a JSON object of field overrides before launching
it; worker processes inherit the environment. When the variable is unset --
always the case under the official framework -- the frozen defaults apply.
"""

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import cast

ENV_VAR = "BFS_AGENT_PARAMS"


@dataclass(frozen=True)
class Params:
    # Value of a target ``d`` steps away is ``value * discount ** d``.
    discount: float = 0.9
    coin_value: float = 1.0
    # Multiplier for coins other than the next one on the planned route.
    off_route: float = 0.5
    # Multiplier for a coin some opponent reaches strictly before we do.
    contested_coin: float = 0.5
    # Value per crate a bomb would destroy (crates may hide coins). Measured
    # against 3 rule_based agents: 0.05-0.15 score alike, 0.3 and up lose
    # about a point by chasing crates over visible coins. 0.15 is the highest
    # of the plateau, which keeps loot-crate demolition fast.
    crate_value: float = 0.15
    # Opponent pressure (``attack``). A bomb that leaves an opponent no escape
    # is worth ``trap_value`` (a kill scores 5); one whose blast merely reaches
    # an opponent that can still escape is worth ``pressure_value``.
    trap_value: float = 4.0
    pressure_value: float = 0.3
    # Pull towards cells whose blast would cover an opponent within reach.
    hunt_value: float = 0.2
    hunt_radius: int = 6
    # Seed for tie-breaking; None draws one from the operating system.
    seed: int | None = None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Params":
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
