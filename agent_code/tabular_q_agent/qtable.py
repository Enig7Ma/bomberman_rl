"""Q-values and visit counts per canonical state, and their file format.

Rows are the encoding's state
indices (only canonical ones are ever touched, see ``symmetry``); columns are
``core.world_model.ACTIONS``. Q-values start at 0.

A table file is a compressed ``.npz`` holding ``q``, ``n`` and ``meta``, a JSON
document with two parts:

- ``header``: what the rows and columns mean -- format version, encoding name,
  fields, radices, action order and the encoding's ``schema_id``. ``load``
  refuses a file whose ``schema_id`` differs from the running encoder's, so a
  feature change can never silently reuse a stale table.
- ``meta``: free-form provenance (configuration, rounds trained, parent
  checkpoint, ...) that the training code fills in.

``save`` is atomic: it writes a temporary file next to the target and renames
it over the target, so a killed training run leaves either the old table or
the new one, never a truncated file.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from .core.world_model import ACTIONS
from .features import Encoding

FORMAT_VERSION: Final = 1
# Table column of each action.
ACTION_COLUMN: Final[dict[str, int]] = {a: i for i, a in enumerate(ACTIONS)}


class SchemaMismatchError(ValueError):
    """The file's rows mean something other than the running encoder's."""


@dataclass
class QTable:
    encoding: Encoding
    q: NDArray[np.float32]
    n: NDArray[np.uint32]
    meta: dict[str, Any] = field(default_factory=dict[str, Any])

    @classmethod
    def zeros(cls, encoding: Encoding) -> "QTable":
        shape = (encoding.n_states, len(ACTIONS))
        return cls(
            encoding,
            np.zeros(shape, dtype=np.float32),
            np.zeros(shape, dtype=np.uint32),
        )

    @property
    def visited_states(self) -> int:
        """States with at least one updated action."""
        return int(np.count_nonzero(self.n.any(axis=1)))

    def header(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "schema_id": self.encoding.schema_id,
            "encoding": self.encoding.name,
            "fields": list(self.encoding.fields),
            "radices": list(self.encoding.radices),
            "actions": list(ACTIONS),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        document = json.dumps({"header": self.header(), "meta": self.meta})
        tmp = path.with_name(f"{path.name}.tmp")
        try:
            # A file object, not a name: numpy would append ``.npz`` to the name.
            with tmp.open("wb") as file:
                np.savez_compressed(file, q=self.q, n=self.n, meta=np.array(document))
                file.flush()
                os.fsync(file.fileno())
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: Path, encoding: Encoding) -> "QTable":
        with np.load(path, allow_pickle=False) as archive:
            document: Any = json.loads(str(archive["meta"].item()))
            q = np.array(archive["q"])
            n = np.array(archive["n"])
        header: Any = document["header"]
        if header.get("schema_id") != encoding.schema_id:
            raise SchemaMismatchError(
                f"{path} was saved for encoding {header.get('encoding')!r} "
                f"(schema {header.get('schema_id')}), but the agent runs "
                f"{encoding.name!r} (schema {encoding.schema_id})"
            )
        shape = (encoding.n_states, len(ACTIONS))
        if q.shape != shape or q.dtype != np.float32:
            raise ValueError(
                f"{path}: q is {q.dtype}{q.shape}, expected float32{shape}"
            )
        if n.shape != shape or n.dtype != np.uint32:
            raise ValueError(f"{path}: n is {n.dtype}{n.shape}, expected uint32{shape}")
        return cls(encoding, q, n, dict(document.get("meta", {})))
