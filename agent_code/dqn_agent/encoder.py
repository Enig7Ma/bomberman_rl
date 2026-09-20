"""Lossless numeric inputs for the DQN (D1: the table's E3 information only).

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
from .features import Extracted, Features, FieldName

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
        self, features: Features, extracted: Extracted | None = None
    ) -> NDArray[np.float32]: ...

    def encode_into(
        self,
        out: NDArray[np.float32],
        features: Features,
        extracted: Extracted | None = None,
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
        self, features: Features, extracted: Extracted | None = None
    ) -> NDArray[np.float32]:
        out = np.empty(self.dim, dtype=np.float32)
        self.encode_into(out, features, extracted)
        return out

    def encode_into(
        self,
        out: NDArray[np.float32],
        features: Features,
        extracted: Extracted | None = None,
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


ENCODERS: Final[dict[str, Encoder]] = {"onehot_e3": OneHotE3()}
