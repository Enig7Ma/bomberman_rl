"""Diagnostics of a network on a fixed, externally supplied set of probe states.

The probe file is built elsewhere and only read here: this module reports the
greedy action, the value level and the churn against the previous measurement,
and refuses a probe whose encoder schema differs from the running one.
"""

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .network import QFunction


def check_q(values: NDArray[np.float32]) -> None:
    if not np.isfinite(values).all() or (np.abs(values) > 50).any():
        raise FloatingPointError("Q is nonfinite or |Q| > 50")


class Probe:
    def __init__(self, x: NDArray[np.float32], masks: NDArray[np.bool_]) -> None:
        if (
            x.ndim != 2
            or x.shape[0] == 0
            or x.dtype != np.float32
            or not np.isfinite(x).all()
        ):
            raise ValueError("expected nonempty finite float32 probe inputs")
        if (
            masks.shape != (len(x), 6)
            or masks.dtype != np.bool_
            or not masks.any(axis=1).all()
        ):
            raise ValueError("expected nonempty bool probe masks")
        self.x, self.masks = x.copy(), masks.copy()
        self.previous: list[int] | None = None
        self.fingerprint = hashlib.sha256(x.tobytes() + masks.tobytes()).hexdigest()

    @classmethod
    def load(cls, path: Path, schema_id: str, dim: int) -> "Probe":
        with np.load(path, allow_pickle=False) as data:
            if str(data["schema_id"].item()) != schema_id or data["x"].shape[1] != dim:
                raise ValueError("probe encoder schema mismatch")
            return cls(data["x"], data["masks"])

    def measure(self, network: QFunction) -> dict[str, Any]:
        q = np.stack([network.values(x) for x in self.x])
        check_q(q)
        masked = np.where(self.masks, q, -np.inf)
        actions = masked.argmax(axis=1)
        top = masked.max(axis=1)
        churn = (
            None
            if self.previous is None
            else float(
                np.count_nonzero(actions != np.array(self.previous)) / len(actions)
            )
        )
        self.previous = actions.tolist()
        return {
            "mean_max_q": float(top.mean()),
            "max_q": float(top.max()),
            "max_abs_q": float(np.abs(q).max()),
            "policy_churn": churn,
            "greedy_histogram": np.bincount(actions, minlength=6).tolist(),
        }
