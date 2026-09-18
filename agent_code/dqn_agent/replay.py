"""Canonical one/multi-step replay, uniform sampling with replacement.

The caller canonicalises x/a and independently x_next/mask_next. This buffer
cannot infer coordinate systems from numeric vectors. Terminal successors are
normalised to zeros (including their potential). Raw sequence reward components
retain discounting and shaping when coefficients change at sampling time.
"""

import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ReplayTransition:
    x: NDArray[np.float32]
    a: int
    base: float
    crates: int
    deaths: int
    phi_unit: float
    phi_unit_next: float
    x_next: NDArray[np.float32]
    mask_next: NDArray[np.bool_]
    done: bool
    stage: int
    round_id: int
    transition_id: int
    # Ordered (base, crates, deaths, phi, phi_next), not pre-discounted.
    reward_steps: tuple[tuple[float, int, int, float, float], ...] = ()


@dataclass(frozen=True)
class ReplayBatch:
    """Owned contiguous arrays; r is computed at sample time."""

    x: NDArray[np.float32]
    a: NDArray[np.uint8]
    base: NDArray[np.float32]
    crates: NDArray[np.uint8]
    deaths: NDArray[np.uint8]
    phi_unit: NDArray[np.float32]
    phi_unit_next: NDArray[np.float32]
    x_next: NDArray[np.float32]
    mask_next: NDArray[np.bool_]
    done: NDArray[np.bool_]
    stage: NDArray[np.uint8]
    round_id: NDArray[np.uint32]
    transition_id: NDArray[np.uint64]
    r: NDArray[np.float32]
    k: NDArray[np.uint8]


def _integer(value: int, name: str, low: int, high: int) -> None:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")


def _finite(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite float32-compatible number")
    if not math.isfinite(value) or abs(value) > float(np.finfo(np.float32).max):
        raise ValueError(f"{name} must be a finite float32-compatible number")


def _array(
    value: object,
    shape: tuple[int, ...],
    dtype: type[np.float32] | type[np.bool_],
    name: str,
) -> None:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array")
    array = cast(NDArray[np.generic], value)
    if array.shape != shape or array.dtype != dtype or not np.isfinite(array).all():
        raise ValueError(
            f"{name} must have shape {shape}, dtype {dtype} and finite values"
        )


def _is_generator(value: object) -> bool:
    return isinstance(value, np.random.Generator)


class ReplayBuffer:
    def __init__(self, capacity: int, input_dim: int) -> None:
        _integer(capacity, "capacity", 1, np.iinfo(np.intp).max)
        _integer(input_dim, "input_dim", 1, np.iinfo(np.intp).max)
        self.input_dim = input_dim
        self.capacity = capacity
        self._size = 0
        self._write = 0
        self._rows: NDArray[np.void] = np.empty(
            capacity,
            dtype=np.dtype(
                [
                    ("x", np.float32, (input_dim,)),
                    ("a", np.uint8),
                    ("base", np.float32),
                    ("crates", np.uint8),
                    ("deaths", np.uint8),
                    ("phi_unit", np.float32),
                    ("phi_unit_next", np.float32),
                    ("x_next", np.float32, (input_dim,)),
                    ("mask_next", np.bool_, (6,)),
                    ("done", np.bool_),
                    ("stage", np.uint8),
                    ("round_id", np.uint32),
                    ("transition_id", np.uint64),
                    ("k", np.uint8),
                    ("components", np.float32, (3, 5)),
                ]
            ),
        )

    def __len__(self) -> int:
        return self._size

    def push(self, transition: ReplayTransition) -> None:
        """Validate before writing; copy inputs; overwrite the oldest when full.

        Integer scalar fields use Python ints, rejecting bools and overflow.
        Arrays must already have the specified float32/bool dtypes.
        """
        t = transition
        _array(t.x, (self.input_dim,), np.float32, "x")
        _array(t.x_next, (self.input_dim,), np.float32, "x_next")
        _array(t.mask_next, (6,), np.bool_, "mask_next")
        if type(t.done) is not bool:
            raise ValueError("done must be bool")
        if not t.done and not t.mask_next.any():
            raise ValueError("nonterminal mask_next must allow an action")
        for name, value, high in (
            ("a", t.a, 5),
            ("crates", t.crates, 255),
            ("deaths", t.deaths, 1),
            ("stage", t.stage, 255),
            ("round_id", t.round_id, 2**32 - 1),
            ("transition_id", t.transition_id, 2**64 - 1),
        ):
            _integer(value, name, 0, high)
        for name, value in (
            ("base", t.base),
            ("phi_unit", t.phi_unit),
            ("phi_unit_next", t.phi_unit_next),
        ):
            _finite(value, name)
        if not 0 <= t.phi_unit <= 1 or not 0 <= t.phi_unit_next <= 1:
            raise ValueError("unit potentials must be in [0, 1]")
        steps = t.reward_steps or (
            (
                t.base,
                t.crates,
                t.deaths,
                t.phi_unit,
                0.0 if t.done else t.phi_unit_next,
            ),
        )
        if not 1 <= len(steps) <= 3:
            raise ValueError("sequence length must be in 1..3")
        components = np.zeros((3, 5), dtype=np.float32)
        for i, (base, crates, deaths, phi, phi_next) in enumerate(steps):
            _integer(crates, "sequence crates", 0, 255)
            _integer(deaths, "sequence deaths", 0, 1)
            for value in (base, phi, phi_next):
                _finite(value, "sequence component")
            if not 0 <= phi <= 1 or not 0 <= phi_next <= 1:
                raise ValueError("invalid sequence potential")
            components[i] = (base, crates, deaths, phi, phi_next)
        if t.done:
            components[len(steps) - 1, 4] = 0
        self._rows[self._write] = (
            t.x,
            t.a,
            t.base,
            t.crates,
            t.deaths,
            t.phi_unit,
            0.0 if t.done else t.phi_unit_next,
            np.zeros_like(t.x_next) if t.done else t.x_next,
            np.zeros_like(t.mask_next) if t.done else t.mask_next,
            t.done,
            t.stage,
            t.round_id,
            t.transition_id,
            len(steps),
            components,
        )
        self._write = (self._write + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        *,
        gamma: float,
        c_coin: float = 0.0,
        crate_aid: float = 0.0,
        death_aid: float = 0.0,
    ) -> ReplayBatch:
        """Uniform independent draws with replacement from filled slots only.

        A batch may exceed len(self); an empty buffer or nonpositive batch is
        rejected. Uses only the supplied Generator, never global RNG state.
        """
        _integer(batch_size, "batch_size", 1, np.iinfo(np.intp).max)
        if not self._size:
            raise ValueError("cannot sample an empty replay buffer")
        if not _is_generator(rng):
            raise ValueError("rng must be a numpy.random.Generator")
        for name, value in (
            ("gamma", gamma),
            ("c_coin", c_coin),
            ("crate_aid", crate_aid),
            ("death_aid", death_aid),
        ):
            _finite(value, name)
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        rows = self._rows[rng.integers(self._size, size=batch_size)]
        # float64 intermediates reduce cancellation; exported rewards are float32.
        reward = (
            rows["base"].astype(np.float64)
            + crate_aid * rows["crates"].astype(np.float64)
            + death_aid * rows["deaths"].astype(np.float64)
            + c_coin
            * (
                gamma * rows["phi_unit_next"].astype(np.float64)
                - rows["phi_unit"].astype(np.float64)
            )
        )
        multi = rows["k"] > 1
        if multi.any():
            c = rows["components"][multi].astype(np.float64)
            per_step = c[:, :, 0] + crate_aid * c[:, :, 1] + death_aid * c[:, :, 2]
            per_step += c_coin * (gamma * c[:, :, 4] - c[:, :, 3])
            reward[multi] = np.sum(per_step * np.power(gamma, np.arange(3)), axis=1)
        if (
            not np.isfinite(reward).all()
            or (np.abs(reward) > np.finfo(np.float32).max).any()
        ):
            raise ValueError("computed rewards exceed finite float32 range")
        return ReplayBatch(
            x=cast(NDArray[np.float32], rows["x"].copy()),
            a=cast(NDArray[np.uint8], rows["a"].copy()),
            base=cast(NDArray[np.float32], rows["base"].copy()),
            crates=cast(NDArray[np.uint8], rows["crates"].copy()),
            deaths=cast(NDArray[np.uint8], rows["deaths"].copy()),
            phi_unit=cast(NDArray[np.float32], rows["phi_unit"].copy()),
            phi_unit_next=cast(NDArray[np.float32], rows["phi_unit_next"].copy()),
            x_next=cast(NDArray[np.float32], rows["x_next"].copy()),
            mask_next=cast(NDArray[np.bool_], rows["mask_next"].copy()),
            done=cast(NDArray[np.bool_], rows["done"].copy()),
            stage=cast(NDArray[np.uint8], rows["stage"].copy()),
            round_id=cast(NDArray[np.uint32], rows["round_id"].copy()),
            transition_id=cast(NDArray[np.uint64], rows["transition_id"].copy()),
            r=reward.astype(np.float32),
            k=cast(NDArray[np.uint8], rows["k"].copy()),
        )

    def snapshot(self) -> dict[str, Any]:
        """Physical ring order, filled portion only; no aliases into storage."""
        return {
            "format_version": 2,
            "capacity": self.capacity,
            "input_dim": self.input_dim,
            "size": self._size,
            "write": self._write,
            "rows": self._rows[: self._size].copy(),
        }

    @classmethod
    def from_snapshot(cls, state: dict[str, Any]) -> "ReplayBuffer":
        result = cls(state["capacity"], state["input_dim"])
        size, write, rows = state["size"], state["write"], state["rows"]
        version = state.get("format_version", 2)
        if version == 1:
            legacy = np.dtype(result._rows.dtype.descr[:-2])
            if rows.dtype != legacy:
                raise ValueError("legacy replay schema mismatch")
            upgraded = np.zeros(size, dtype=result._rows.dtype)
            for name in legacy.names or ():
                upgraded[name] = rows[name]
            upgraded["k"] = 1
            for column, name in enumerate(
                ("base", "crates", "deaths", "phi_unit", "phi_unit_next")
            ):
                upgraded["components"][:, 0, column] = rows[name]
            rows = upgraded
        elif version != 2:
            raise ValueError("unsupported replay format")
        _integer(size, "size", 0, result.capacity)
        _integer(write, "write", 0, result.capacity - 1)
        if size < result.capacity and write != size:
            raise ValueError("invalid partial ring write position")
        if rows.dtype != result._rows.dtype or rows.shape != (size,):
            raise ValueError("replay schema mismatch")
        if ((rows["k"] < 1) | (rows["k"] > 3)).any():
            raise ValueError("invalid sequence length")
        c = rows["components"]
        if not np.isfinite(c).all() or ((c[:, :, 3:] < 0) | (c[:, :, 3:] > 1)).any():
            raise ValueError("invalid sequence components")
        counts = c[:, :, 1:3]
        if (
            (counts < 0).any()
            or (counts != np.floor(counts)).any()
            or (c[:, :, 1] > 255).any()
            or (c[:, :, 2] > 1).any()
        ):
            raise ValueError("invalid sequence event counts")
        for i in range(size):
            k = int(rows["k"][i])
            if c[i, k:].any() or (rows["done"][i] and c[i, k - 1, 4] != 0):
                raise ValueError("invalid sequence padding/terminal potential")
            if (
                any(
                    c[i, 0, j] != rows[name][i]
                    for j, name in enumerate(("base", "crates", "deaths", "phi_unit"))
                )
                or c[i, k - 1, 4] != rows["phi_unit_next"][i]
            ):
                raise ValueError("inconsistent sequence components")
            if not np.array_equal(c[i, : k - 1, 4], c[i, 1:k, 3]):
                raise ValueError("discontinuous sequence potentials")
        for name in ("x", "x_next", "base", "phi_unit", "phi_unit_next"):
            if not np.isfinite(rows[name]).all():
                raise ValueError("nonfinite replay data")
        if (rows["a"] > 5).any() or (rows["deaths"] > 1).any():
            raise ValueError("invalid replay action/death")
        for name in ("phi_unit", "phi_unit_next"):
            if ((rows[name] < 0) | (rows[name] > 1)).any():
                raise ValueError("invalid replay potential")
        terminal = rows["done"]
        if (
            rows["mask_next"][terminal].any()
            or rows["x_next"][terminal].any()
            or rows["phi_unit_next"][terminal].any()
        ):
            raise ValueError("invalid terminal replay successor")
        if not rows["mask_next"][~terminal].any(axis=1).all():
            raise ValueError("empty live replay mask")
        result._rows[:size] = rows
        result._size, result._write = size, write
        return result
