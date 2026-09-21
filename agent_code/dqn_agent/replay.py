"""Canonical one/multi-step replay, uniform sampling with replacement.

The caller canonicalises x/a and independently x_next/mask_next. This buffer
cannot infer coordinate systems from numeric vectors. Terminal successors are
normalised to zeros (including their potential). Raw sequence reward components
retain discounting and shaping when coefficients change at sampling time.
"""

import math
from dataclasses import dataclass
from typing import Any, Final, cast

import numpy as np
from numpy.typing import NDArray

# One row's reward ingredients, per step of the (1..3)-step sequence:
# base score delta, crates destroyed, deaths, coin potential before and after,
# crates a confirmed bomb booked, spot potential before and after, and the
# attack category of a confirmed bomb drop, and the opponent potential before
# and after.
COMPONENT_NAMES: Final = (
    "base",
    "crates",
    "deaths",
    "phi_unit",
    "phi_unit_next",
    "bombs",
    "spot_unit",
    "spot_unit_next",
    "attacks",
    "hunt_unit",
    "hunt_unit_next",
)
COMPONENTS: Final = len(COMPONENT_NAMES)
FORMAT_VERSION: Final = 5


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
    # Live crates a confirmed bomb drop booked, for the ``bomb_aid`` anchor; 0
    # on every other action.
    bombs: int = 0
    # ``1 / (1 + d)`` on the distance to the best bombing spot, the second
    # potential; times ``spot_potential`` at sample time.
    spot_unit: float = 0.0
    spot_unit_next: float = 0.0
    # 0 none, 1 pressure, 2 trap, and 0 unless a bomb was actually dropped.
    attacks: int = 0
    # ``1 / (1 + d)`` on the distance to the nearest opponent; times
    # ``hunt_potential`` at sample time.
    hunt_unit: float = 0.0
    hunt_unit_next: float = 0.0
    # Ordered COMPONENTS per step, not pre-discounted.
    reward_steps: tuple[
        tuple[float, int, int, float, float, int, float, float, int, float, float],
        ...,
    ] = ()


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
    bombs: NDArray[np.uint8]
    spot_unit: NDArray[np.float32]
    spot_unit_next: NDArray[np.float32]
    attacks: NDArray[np.uint8]
    hunt_unit: NDArray[np.float32]
    hunt_unit_next: NDArray[np.float32]
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
                    ("bombs", np.uint8),
                    ("spot_unit", np.float32),
                    ("spot_unit_next", np.float32),
                    ("attacks", np.uint8),
                    ("hunt_unit", np.float32),
                    ("hunt_unit_next", np.float32),
                    ("x_next", np.float32, (input_dim,)),
                    ("mask_next", np.bool_, (6,)),
                    ("done", np.bool_),
                    ("stage", np.uint8),
                    ("round_id", np.uint32),
                    ("transition_id", np.uint64),
                    ("k", np.uint8),
                    ("components", np.float32, (3, COMPONENTS)),
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
            ("bombs", t.bombs, 255),
            ("attacks", t.attacks, 2),
            ("stage", t.stage, 255),
            ("round_id", t.round_id, 2**32 - 1),
            ("transition_id", t.transition_id, 2**64 - 1),
        ):
            _integer(value, name, 0, high)
        for name, value in (
            ("base", t.base),
            ("phi_unit", t.phi_unit),
            ("phi_unit_next", t.phi_unit_next),
            ("spot_unit", t.spot_unit),
            ("spot_unit_next", t.spot_unit_next),
            ("hunt_unit", t.hunt_unit),
            ("hunt_unit_next", t.hunt_unit_next),
        ):
            _finite(value, name)
        for name in (
            "phi_unit",
            "phi_unit_next",
            "spot_unit",
            "spot_unit_next",
            "hunt_unit",
            "hunt_unit_next",
        ):
            if not 0 <= getattr(t, name) <= 1:
                raise ValueError("unit potentials must be in [0, 1]")
        steps = t.reward_steps or (
            (
                t.base,
                t.crates,
                t.deaths,
                t.phi_unit,
                0.0 if t.done else t.phi_unit_next,
                t.bombs,
                t.spot_unit,
                0.0 if t.done else t.spot_unit_next,
                t.attacks,
                t.hunt_unit,
                0.0 if t.done else t.hunt_unit_next,
            ),
        )
        if not 1 <= len(steps) <= 3:
            raise ValueError("sequence length must be in 1..3")
        components = np.zeros((3, COMPONENTS), dtype=np.float32)
        for i, step in enumerate(steps):
            if len(step) != COMPONENTS:
                raise ValueError(f"a sequence step has {COMPONENTS} components")
            (
                base,
                crates,
                deaths,
                phi,
                phi_next,
                bombs,
                spot,
                spot_next,
                attacks,
                hunt,
                hunt_next,
            ) = step
            _integer(crates, "sequence crates", 0, 255)
            _integer(deaths, "sequence deaths", 0, 1)
            _integer(bombs, "sequence bombs", 0, 255)
            _integer(attacks, "sequence attacks", 0, 2)
            for value in (base, phi, phi_next, spot, spot_next, hunt, hunt_next):
                _finite(value, "sequence component")
            if any(
                not 0 <= v <= 1
                for v in (phi, phi_next, spot, spot_next, hunt, hunt_next)
            ):
                raise ValueError("invalid sequence potential")
            components[i] = step
        if t.done:
            for column in (4, 7, 10):
                components[len(steps) - 1, column] = 0
        self._rows[self._write] = (
            t.x,
            t.a,
            t.base,
            t.crates,
            t.deaths,
            t.phi_unit,
            0.0 if t.done else t.phi_unit_next,
            t.bombs,
            t.spot_unit,
            0.0 if t.done else t.spot_unit_next,
            t.attacks,
            t.hunt_unit,
            0.0 if t.done else t.hunt_unit_next,
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
        bomb_aid: float = 0.0,
        spot_potential: float = 0.0,
        attack_aid: float = 0.0,
        hunt_potential: float = 0.0,
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
            ("bomb_aid", bomb_aid),
            ("spot_potential", spot_potential),
            ("attack_aid", attack_aid),
            ("hunt_potential", hunt_potential),
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
            + bomb_aid * rows["bombs"].astype(np.float64)
            + attack_aid * rows["attacks"].astype(np.float64)
            + c_coin
            * (
                gamma * rows["phi_unit_next"].astype(np.float64)
                - rows["phi_unit"].astype(np.float64)
            )
            + spot_potential
            * (
                gamma * rows["spot_unit_next"].astype(np.float64)
                - rows["spot_unit"].astype(np.float64)
            )
            + hunt_potential
            * (
                gamma * rows["hunt_unit_next"].astype(np.float64)
                - rows["hunt_unit"].astype(np.float64)
            )
        )
        multi = rows["k"] > 1
        if multi.any():
            c = rows["components"][multi].astype(np.float64)
            per_step = (
                c[:, :, 0]
                + crate_aid * c[:, :, 1]
                + death_aid * c[:, :, 2]
                + bomb_aid * c[:, :, 5]
                + attack_aid * c[:, :, 8]
            )
            per_step += c_coin * (gamma * c[:, :, 4] - c[:, :, 3])
            per_step += spot_potential * (gamma * c[:, :, 7] - c[:, :, 6])
            per_step += hunt_potential * (gamma * c[:, :, 10] - c[:, :, 9])
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
            bombs=cast(NDArray[np.uint8], rows["bombs"].copy()),
            spot_unit=cast(NDArray[np.float32], rows["spot_unit"].copy()),
            spot_unit_next=cast(NDArray[np.float32], rows["spot_unit_next"].copy()),
            attacks=cast(NDArray[np.uint8], rows["attacks"].copy()),
            hunt_unit=cast(NDArray[np.float32], rows["hunt_unit"].copy()),
            hunt_unit_next=cast(NDArray[np.float32], rows["hunt_unit_next"].copy()),
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
            "format_version": FORMAT_VERSION,
            "capacity": self.capacity,
            "input_dim": self.input_dim,
            "size": self._size,
            "write": self._write,
            "rows": self._rows[: self._size].copy(),
        }

    def _legacy_dtype(self, version: int) -> np.dtype[np.void]:
        """The row layout of an older file: before the attack column (3),
        before the bomb/spot columns as well (2), and before the n-step
        ``k``/``components`` columns on top of that (1)."""
        dropped: set[str] = {"hunt_unit", "hunt_unit_next"}
        if version != 4:
            dropped |= {"attacks"}
        if version not in (3, 4):
            dropped |= {"bombs", "spot_unit", "spot_unit_next"}
        fields = [entry for entry in self._rows.dtype.descr if entry[0] not in dropped]
        if version in (3, 4):
            fields[-1] = ("components", "<f4", (3, 8 if version == 3 else 9))
            return np.dtype(fields)
        if version == 2:
            fields[-1] = ("components", "<f4", (3, 5))
            return np.dtype(fields)
        return np.dtype(fields[:-2])

    def _upgrade(
        self, rows: NDArray[np.void], size: int, version: int
    ) -> NDArray[np.void]:
        """Read an older file into the current layout: no attack aid, for
        versions 1 and 2 no bombs and no spot potential either, and for
        version 1 a one-step sequence per row."""
        legacy = self._legacy_dtype(version)
        if rows.dtype != legacy:
            raise ValueError("legacy replay schema mismatch")
        upgraded = np.zeros(size, dtype=self._rows.dtype)
        for name in legacy.names or ():
            if name != "components":
                upgraded[name] = rows[name]
        if version in (2, 3, 4):
            width = {2: 5, 3: 8, 4: 9}[version]
            upgraded["components"][:, :, :width] = rows["components"]
            return upgraded
        upgraded["k"] = 1
        for column, name in enumerate(
            ("base", "crates", "deaths", "phi_unit", "phi_unit_next")
        ):
            upgraded["components"][:, 0, column] = rows[name]
        return upgraded

    @classmethod
    def from_snapshot(cls, state: dict[str, Any]) -> "ReplayBuffer":
        result = cls(state["capacity"], state["input_dim"])
        size, write, rows = state["size"], state["write"], state["rows"]
        version = state.get("format_version", FORMAT_VERSION)
        if version in (1, 2, 3, 4):
            rows = result._upgrade(rows, size, version)
        elif version != FORMAT_VERSION:
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
        potentials = c[:, :, [3, 4, 6, 7, 9, 10]]
        if not np.isfinite(c).all() or ((potentials < 0) | (potentials > 1)).any():
            raise ValueError("invalid sequence components")
        counts = c[:, :, [1, 2, 5, 8]]
        if (
            (counts < 0).any()
            or (counts != np.floor(counts)).any()
            or (c[:, :, 1] > 255).any()
            or (c[:, :, 2] > 1).any()
            or (c[:, :, 5] > 255).any()
            or (c[:, :, 8] > 2).any()
        ):
            raise ValueError("invalid sequence event counts")
        for i in range(size):
            k = int(rows["k"][i])
            terminal_tail = rows["done"][i] and any(
                c[i, k - 1, column] != 0 for column in (4, 7, 10)
            )
            if c[i, k:].any() or terminal_tail:
                raise ValueError("invalid sequence padding/terminal potential")
            head = ("base", "crates", "deaths", "phi_unit")
            if (
                any(c[i, 0, j] != rows[name][i] for j, name in enumerate(head))
                or c[i, 0, 5] != rows["bombs"][i]
                or c[i, 0, 8] != rows["attacks"][i]
                or c[i, 0, 6] != rows["spot_unit"][i]
                or c[i, 0, 9] != rows["hunt_unit"][i]
                or c[i, k - 1, 10] != rows["hunt_unit_next"][i]
                or c[i, k - 1, 4] != rows["phi_unit_next"][i]
                or c[i, k - 1, 7] != rows["spot_unit_next"][i]
            ):
                raise ValueError("inconsistent sequence components")
            if any(
                not np.array_equal(c[i, : k - 1, nxt], c[i, 1:k, cur])
                for cur, nxt in ((3, 4), (6, 7), (9, 10))
            ):
                raise ValueError("discontinuous sequence potentials")
        for name in ("x", "x_next", "base", "phi_unit", "phi_unit_next"):
            if not np.isfinite(rows[name]).all():
                raise ValueError("nonfinite replay data")
        if (
            (rows["a"] > 5).any()
            or (rows["deaths"] > 1).any()
            or (rows["attacks"] > 2).any()
        ):
            raise ValueError("invalid replay action/death/attack")
        for name in (
            "phi_unit",
            "phi_unit_next",
            "spot_unit",
            "spot_unit_next",
            "hunt_unit",
            "hunt_unit_next",
        ):
            if ((rows[name] < 0) | (rows[name] > 1)).any():
                raise ValueError("invalid replay potential")
        terminal = rows["done"]
        if (
            rows["mask_next"][terminal].any()
            or rows["x_next"][terminal].any()
            or rows["phi_unit_next"][terminal].any()
            or rows["spot_unit_next"][terminal].any()
            or rows["hunt_unit_next"][terminal].any()
        ):
            raise ValueError("invalid terminal replay successor")
        if not rows["mask_next"][~terminal].any(axis=1).all():
            raise ValueError("empty live replay mask")
        result._rows[:size] = rows
        result._size, result._write = size, write
        return result
