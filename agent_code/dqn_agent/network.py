"""NumPy-only Q-function and portable, validated inference weights."""

import json
import math
import os
import random
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from .core.world_model import ACTIONS
from .encoder import Encoder

FORMAT_VERSION: Final = 1
ARRAY_NAMES: Final = ("W0", "b0", "W1", "b1", "W2", "b2")


class QFunction(Protocol):
    def values(self, x: NDArray[np.float32]) -> NDArray[np.float32]: ...


class ModelFormatError(ValueError):
    """A weight file is invalid or incompatible with the current agent."""


def masked_greedy(
    values: NDArray[np.float32], allowed: Sequence[int], rng: random.Random
) -> int:
    if values.shape != (len(ACTIONS),) or not allowed:
        raise ValueError("expected six Q-values and a nonempty action mask")
    if any(a < 0 or a >= len(ACTIONS) for a in allowed):
        raise ValueError("action index outside 0..5")
    if any(not math.isfinite(float(values[a])) for a in allowed):
        raise ValueError("allowed Q-values must be finite")
    top = max(float(values[a]) for a in allowed)
    return rng.choice([a for a in allowed if float(values[a]) == top])


def _header(encoder: Encoder) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "encoder": encoder.name,
        "schema_id": encoder.schema_id,
        "layer_sizes": [encoder.dim, 128, 128, len(ACTIONS)],
        "activation": "relu",
        "actions": list(ACTIONS),
        "dtype": "float32",
    }


def _check_meta(meta: object) -> dict[str, Any]:
    if not isinstance(meta, dict):
        raise ModelFormatError("metadata must be a JSON object")
    result = cast(dict[str, Any], meta)
    if not isinstance(result.get("config"), dict):
        raise ModelFormatError("metadata config must be an object")
    for name in ("transitions", "updates", "rounds_trained"):
        value: object = result.get(name)
        if type(value) is not int or value < 0:
            raise ModelFormatError(f"metadata {name} must be a nonnegative integer")
    if not isinstance(result.get("stage"), str):
        raise ModelFormatError("metadata stage must be a string")
    for name in ("parent_checkpoint", "git_commit"):
        if name not in result or (
            result[name] is not None and not isinstance(result[name], str)
        ):
            raise ModelFormatError(f"metadata {name} must be a string or null")
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ModelFormatError("metadata must contain finite JSON values") from error
    return result


class QNetwork:
    def __init__(
        self,
        encoder: Encoder,
        arrays: Mapping[str, NDArray[np.float32]],
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        self.header = _header(encoder)
        self.input_dim = encoder.dim
        self.arrays = dict(arrays)
        self.meta: dict[str, Any] = {
            "config": {},
            "transitions": 0,
            "updates": 0,
            "rounds_trained": 0,
            "parent_checkpoint": None,
            "stage": "untrained",
            "git_commit": None,
            **(meta or {}),
        }
        self._validate()

    def _validate(self) -> None:
        if set(self.arrays) != set(ARRAY_NAMES):
            raise ModelFormatError("expected exactly W0,b0,W1,b1,W2,b2")
        sizes = (self.input_dim, 128, 128, len(ACTIONS))
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
            for key, shape in ((f"W{i}", (n_out, n_in)), (f"b{i}", (n_out,))):
                array = self.arrays[key]
                if array.shape != shape or array.dtype != np.dtype(np.float32):
                    raise ModelFormatError(f"{key} must be float32 with shape {shape}")
                if not np.isfinite(array).all():
                    raise ModelFormatError(f"{key} contains non-finite weights")
        _check_meta(self.meta)

    @classmethod
    def random(cls, encoder: Encoder, seed: int | None) -> "QNetwork":
        """Local RNG, uniform +/-1/sqrt(fan_in), like a Linear initialisation."""
        rng = np.random.default_rng(seed)
        sizes = (encoder.dim, 128, 128, len(ACTIONS))
        arrays: dict[str, NDArray[np.float32]] = {}
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
            bound = 1.0 / math.sqrt(n_in)
            arrays[f"W{i}"] = rng.uniform(-bound, bound, (n_out, n_in)).astype(
                np.float32
            )
            arrays[f"b{i}"] = rng.uniform(-bound, bound, n_out).astype(np.float32)
        return cls(encoder, arrays, {"init_seed": seed})

    def values(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        if x.shape != (self.input_dim,) or x.dtype != np.dtype(np.float32):
            raise ValueError(f"expected float32 input of shape ({self.input_dim},)")
        if not np.isfinite(x).all():
            raise ValueError("network input must be finite")
        h = x
        for i in range(3):
            h = self.arrays[f"W{i}"] @ h + self.arrays[f"b{i}"]
            if i < 2:
                np.maximum(h, np.float32(0.0), out=h)
        if not np.isfinite(h).all():
            raise ValueError("network produced non-finite Q-values")
        return h

    def save(self, path: Path) -> None:
        self._validate()
        document = json.dumps(
            {"header": self.header, "meta": self.meta}, allow_nan=False
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as file:
                temporary = Path(file.name)
                np.savez_compressed(
                    file, allow_pickle=False, **self.arrays, meta=np.array(document)
                )
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path, encoder: Encoder) -> "QNetwork":
        with np.load(path, allow_pickle=False) as archive:
            if sorted(archive.files) != sorted((*ARRAY_NAMES, "meta")):
                raise ModelFormatError("unexpected archive entries")
            raw = archive["meta"]
            if raw.shape != () or raw.dtype.kind != "U":
                raise ModelFormatError("meta must be a scalar JSON string")
            document: Any = json.loads(str(raw.item()))
            if not isinstance(document, dict):
                raise ModelFormatError("expected a JSON object")
            document = cast(dict[str, Any], document)
            if set(document) != {"header", "meta"}:
                raise ModelFormatError("expected header and meta objects")
            # JSON comparison also distinguishes True from 1 and 128.0 from 128.
            if json.dumps(document["header"], sort_keys=True) != json.dumps(
                _header(encoder), sort_keys=True
            ):
                raise ModelFormatError("incompatible network header or schema_id")
            meta = _check_meta(document["meta"])
            arrays = {name: archive[name].copy() for name in ARRAY_NAMES}
        return cls(encoder, arrays, meta)
