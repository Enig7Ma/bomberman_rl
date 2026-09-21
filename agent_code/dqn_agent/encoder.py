"""Lossless numeric inputs for the DQN, carrying the table's E3 information only.

Canonicalisation belongs to the caller: ``index, g = canonical(f, E3)`` then
``encode(E3.decode(index), extracted)``. The encoder does not choose a symmetry
or transform actions. OneHotE3 ignores ``extracted`` so no extra information
leaks into the strict table-versus-network comparison.
"""

import hashlib
import json
from typing import Final, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from . import features as feature_defs
from .core.world_model import ACTIONS
from .features import Extracted, Features, FieldName
from .symmetry import IDENTITY, Symmetry

BlockKind = Literal["bits", "onehot"]
# Ordered blocks, their representation and width; offsets are cumulative.
BLOCKS: Final[tuple[tuple[FieldName, BlockKind, int], ...]] = (
    ("mask", "bits", 6),
    ("coin_dir", "onehot", 6),
    ("crate_dir", "onehot", 6),
    ("bomb_yield", "onehot", 4),
    ("danger", "bits", 1),
    ("opp_dir", "onehot", 6),
    ("attack", "onehot", 3),
)


class Encoder(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    @property
    def schema_id(self) -> str: ...

    def encode(
        self,
        features: Features,
        extracted: Extracted | None = None,
        symmetry: Symmetry = IDENTITY,
    ) -> NDArray[np.float32]: ...

    def encode_into(
        self,
        out: NDArray[np.float32],
        features: Features,
        extracted: Extracted | None = None,
        symmetry: Symmetry = IDENTITY,
    ) -> None: ...


def _category(value: object, limit: int, field: str) -> int:
    if not isinstance(value, int) or not 0 <= value < limit:
        raise ValueError(f"{field} must be an integer in 0..{limit - 1}, got {value!r}")
    return value


class OneHotE3:
    @property
    def name(self) -> str:
        return "onehot_e3"

    @property
    def dim(self) -> int:
        return sum(width for _, _, width in BLOCKS)

    @property
    def schema_id(self) -> str:
        schema = {
            "name": self.name,
            "dim": self.dim,
            "blocks": BLOCKS,
            "dtype": "float32",
            "feature_schema": feature_defs.ENCODINGS["E3"].schema_id,
        }
        digest = hashlib.sha256(json.dumps(schema, sort_keys=True).encode())
        return digest.hexdigest()[:16]

    def encode(
        self,
        features: Features,
        extracted: Extracted | None = None,
        symmetry: Symmetry = IDENTITY,
    ) -> NDArray[np.float32]:
        out = np.empty(self.dim, dtype=np.float32)
        self.encode_into(out, features, extracted, symmetry)
        return out

    def encode_into(
        self,
        out: NDArray[np.float32],
        features: Features,
        extracted: Extracted | None = None,
        symmetry: Symmetry = IDENTITY,
    ) -> None:
        """Overwrite a float32 buffer; allocate no output or temporary arrays."""
        self._check_vector(out)
        out.fill(0.0)
        offset = 0
        for field, kind, width in BLOCKS:
            limit = (1 << width) if kind == "bits" else width
            value = _category(getattr(features, field), limit, field)
            if kind == "bits":
                for bit in range(width):
                    out[offset + bit] = (value >> bit) & 1
            else:
                out[offset + value] = 1.0
            offset += width

    def decode(self, vector: NDArray[np.float32]) -> Features:
        """Invert a valid encoded vector; reject fractional or ambiguous blocks."""
        self._check_vector(vector)
        if not np.all((vector == 0.0) | (vector == 1.0)):
            raise ValueError("encoded values must be binary")
        values: dict[FieldName, int] = {}
        offset = 0
        for field, kind, width in BLOCKS:
            if kind == "bits":
                values[field] = sum(
                    int(vector[offset + bit]) << bit for bit in range(width)
                )
            else:
                block = vector[offset : offset + width]
                if np.count_nonzero(block) != 1:
                    raise ValueError(f"{field} must have exactly one active value")
                values[field] = int(np.argmax(block))
            offset += width
        return Features(**values)

    def _check_vector(self, vector: NDArray[np.float32]) -> None:
        if vector.shape != (self.dim,) or vector.dtype != np.dtype(np.float32):
            raise ValueError(f"expected float32 vector of shape ({self.dim},)")


# --- dense_v1 ---------------------------------------------------------------
# The representation experiment of dev/dqn.md D9. E3 stores *which way* to go;
# the tabular log's last finding is that what now binds is the representation,
# not the learning: a direction without a distance cannot tell "a coin around
# the corner" from "a coin fifteen steps away", and one mask bit cannot tell a
# cell with one escape from a cell with eight.
#
# dense_v1 keeps the whole 32-d one-hot as a prefix, so it is a strict superset
# of the table's information, and appends the numbers behind those categories.
# Guard rail (rules.pdf 7): no block names an action to take. Per-move tiers
# are the mask in graded form and the option bits are the ``*_dir`` features
# without the random tie-break, both already in E3 in coarser form.

# Per move: legal, safety tier, escape room, and whether the coin, crate-spot
# and opponent directions point that way.
MOVE_FEATURES: Final = 6
# WAIT and BOMB: legal, safety tier, escape room.
FIXED_FEATURES: Final = 3
# Scale of each distance block; beyond it a distance is "far".
FAR: Final = 16.0
MOVES: Final = 4
DISTANCE_FIELDS: Final = ("coin", "crate", "opponent", "hunt")
SCALARS: Final = (
    "bomb_hits",
    "coins_visible",
    "opponents_alive",
    "crates_left",
    "step",
    "danger",
    "lethal_delay",
    "lethal_none",
)


def _unit(value: float, scale: float) -> float:
    return min(float(value), scale) / scale


class DenseV1:
    """``OneHotE3`` plus graded per-action and distance blocks.

    The per-move blocks are indexed by the *canonical* direction, so the input
    is canonical exactly like the one-hot prefix: the caller passes the
    symmetry that ``canonical`` returned and every direction-indexed value is
    written to the slot of its image.
    """

    @property
    def name(self) -> str:
        return "dense_v1"

    @property
    def dim(self) -> int:
        return (
            OneHotE3().dim
            + MOVES * MOVE_FEATURES
            + 2 * FIXED_FEATURES
            + 2 * len(DISTANCE_FIELDS)
            + len(SCALARS)
        )

    @property
    def schema_id(self) -> str:
        schema = {
            "name": self.name,
            "dim": self.dim,
            "blocks": BLOCKS,
            "move_features": MOVE_FEATURES,
            "fixed_features": FIXED_FEATURES,
            "distances": DISTANCE_FIELDS,
            "scalars": SCALARS,
            "far": FAR,
            "dtype": "float32",
            "feature_schema": feature_defs.ENCODINGS["E3"].schema_id,
        }
        digest = hashlib.sha256(json.dumps(schema, sort_keys=True).encode())
        return digest.hexdigest()[:16]

    def encode(
        self,
        features: Features,
        extracted: Extracted | None = None,
        symmetry: Symmetry = IDENTITY,
    ) -> NDArray[np.float32]:
        out = np.empty(self.dim, dtype=np.float32)
        self.encode_into(out, features, extracted, symmetry)
        return out

    def encode_into(
        self,
        out: NDArray[np.float32],
        features: Features,
        extracted: Extracted | None = None,
        symmetry: Symmetry = IDENTITY,
    ) -> None:
        if out.shape != (self.dim,) or out.dtype != np.dtype(np.float32):
            raise ValueError(f"expected float32 vector of shape ({self.dim},)")
        if extracted is None:
            raise ValueError("dense_v1 needs the step's Extracted")
        prefix = OneHotE3()
        prefix.encode_into(out[: prefix.dim], features)
        out[prefix.dim :] = 0.0
        offset = prefix.dim
        options = extracted.options
        for raw in range(MOVES):
            block = offset + symmetry.direction(raw) * MOVE_FEATURES
            tier = extracted.tiers[raw] if extracted.tiers else -1
            refuges = extracted.refuges[raw] if extracted.refuges else 0
            out[block] = float(tier >= 0)
            out[block + 1] = max(tier, 0) / 4.0
            out[block + 2] = _unit(refuges, 8.0)
            for i, field in enumerate(("coin_dir", "crate_dir", "opp_dir")):
                out[block + 3 + i] = float(raw in options.get(field, frozenset()))
        offset += MOVES * MOVE_FEATURES
        for i, action in enumerate(ACTIONS[MOVES:]):
            index = ACTIONS.index(action)
            tier = extracted.tiers[index] if extracted.tiers else -1
            refuges = extracted.refuges[index] if extracted.refuges else 0
            block = offset + i * FIXED_FEATURES
            out[block] = float(tier >= 0)
            out[block + 1] = max(tier, 0) / 4.0
            out[block + 2] = _unit(refuges, 8.0)
        offset += 2 * FIXED_FEATURES
        distances = (
            extracted.coin_distance,
            extracted.crate_distance,
            extracted.opponent_distance,
            extracted.hunt_distance,
        )
        for i, distance in enumerate(distances):
            out[offset + 2 * i] = 1.0 if distance is None else _unit(distance, FAR)
            out[offset + 2 * i + 1] = float(distance is None)
        offset += 2 * len(DISTANCE_FIELDS)
        lethal = extracted.lethal_offsets
        out[offset] = _unit(extracted.bomb_hits, 6.0)
        out[offset + 1] = _unit(extracted.coins_visible, 9.0)
        out[offset + 2] = _unit(extracted.opponents_alive, 3.0)
        out[offset + 3] = _unit(extracted.crates_left, 130.0)
        out[offset + 4] = _unit(extracted.step, 400.0)
        out[offset + 5] = float(bool(lethal))
        out[offset + 6] = _unit(lethal[0], 4.0) if lethal else 1.0
        out[offset + 7] = float(not lethal)


class DenseV2(DenseV1):
    """``dense_v1`` plus the direction to the nearest opponent at any distance.

    Measured on the shipped `dense_v1` agent: **37%** of its decisions happen
    after the last coin and the last crate are gone, and in **51%** of those
    `opp_dir` is NONE, because it only reports hunt cells within
    ``hunt_radius`` (6) and the mean distance to an opponent is 6.9. With no
    coin, no crate and no hunt direction, every direction-bearing input is zero
    and the agent random-walks: 140 distinct cells and 745 revisits over six
    rounds, with an almost uniform action distribution.

    This adds the one thing that is still knowable in those states -- which way
    the nearest opponent is -- as four direction bits permuted with the rest.
    The distance to that opponent is already in `dense_v1`; what was missing
    was the direction to go with it.
    """

    @property
    def name(self) -> str:
        return "dense_v2"

    @property
    def dim(self) -> int:
        return super().dim + MOVES

    @property
    def schema_id(self) -> str:
        schema = {"base": super().schema_id, "approach": MOVES, "name": self.name}
        digest = hashlib.sha256(json.dumps(schema, sort_keys=True).encode())
        return digest.hexdigest()[:16]

    def encode_into(
        self,
        out: NDArray[np.float32],
        features: Features,
        extracted: Extracted | None = None,
        symmetry: Symmetry = IDENTITY,
    ) -> None:
        if out.shape != (self.dim,) or out.dtype != np.dtype(np.float32):
            raise ValueError(f"expected float32 vector of shape ({self.dim},)")
        if extracted is None:
            raise ValueError("dense_v2 needs the step's Extracted")
        base = DenseV1()
        base.encode_into(out[: base.dim], features, extracted, symmetry)
        out[base.dim :] = 0.0
        for raw in extracted.approach_options:
            if raw < MOVES:
                out[base.dim + symmetry.direction(raw)] = 1.0


# dense_v3's coin-race block: how the visible coins divide between us and the
# opponents, and which way the nearest one we would actually win lies.
RACE_SCALARS: Final = (
    "own_coin_distance",
    "own_coin_none",
    "owned_coins",
    "contested_coins",
    "coins_within_5",
    "coins_within_10",
)


class DenseV3(DenseV2):
    """``dense_v2`` plus who wins the race for each coin.

    Measured over real rounds: in a third of the steps where a coin is visible
    at least one of them is reached sooner by an opponent, and in **23-32%** of
    those steps the nearest coin the agent would actually *win* is not the
    nearest coin at all. `coin_dir` points at the nearest one regardless, so
    until now the agent walked towards coins it was going to lose, which is
    precisely how it lost duels: in its lost rounds it took 3.26 coins to
    `bfs_agent`'s 5.74.
    """

    @property
    def name(self) -> str:
        return "dense_v3"

    @property
    def dim(self) -> int:
        return super().dim + MOVES + len(RACE_SCALARS)

    @property
    def schema_id(self) -> str:
        schema = {"base": super().schema_id, "race": RACE_SCALARS, "name": self.name}
        digest = hashlib.sha256(json.dumps(schema, sort_keys=True).encode())
        return digest.hexdigest()[:16]

    def encode_into(
        self,
        out: NDArray[np.float32],
        features: Features,
        extracted: Extracted | None = None,
        symmetry: Symmetry = IDENTITY,
    ) -> None:
        if out.shape != (self.dim,) or out.dtype != np.dtype(np.float32):
            raise ValueError(f"expected float32 vector of shape ({self.dim},)")
        if extracted is None:
            raise ValueError("dense_v3 needs the step's Extracted")
        base = DenseV2()
        base.encode_into(out[: base.dim], features, extracted, symmetry)
        offset = base.dim
        out[offset:] = 0.0
        for raw in extracted.own_coin_options:
            if raw < MOVES:
                out[offset + symmetry.direction(raw)] = 1.0
        offset += MOVES
        distance = extracted.own_coin_distance
        out[offset] = 1.0 if distance is None else _unit(distance, FAR)
        out[offset + 1] = float(distance is None)
        out[offset + 2] = _unit(extracted.owned_coins, 9.0)
        out[offset + 3] = _unit(extracted.contested_coins, 9.0)
        out[offset + 4] = _unit(extracted.coins_within_5, 9.0)
        out[offset + 5] = _unit(extracted.coins_within_10, 9.0)


ENCODERS: Final[dict[str, Encoder]] = {
    "onehot_e3": OneHotE3(),
    "dense_v1": DenseV1(),
    "dense_v2": DenseV2(),
    "dense_v3": DenseV3(),
}
