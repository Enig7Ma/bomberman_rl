"""Atomic individual files; generation checks prevent silent mixed resumes.

Saving is round-boundary only. Replay is written before checkpoint, then the
NumPy export. Interrupted multi-file saves may require intervention; there is
no claim of a cross-file atomic transaction. Older replay is explicit degraded
resume, newer/unrelated/missing replay is rejected.
"""

import json
import os
import pickle
import tempfile
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any, BinaryIO, cast

import numpy as np
import torch

from .replay import ReplayBuffer


def atomic_write(path: Path, writer: Callable[[BinaryIO], None]) -> float:
    started = perf_counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, delete=False, suffix=".tmp"
        ) as file:
            temporary = Path(file.name)
            writer(cast(BinaryIO, file))
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return perf_counter() - started


def save_replay(
    path: Path,
    replay: ReplayBuffer,
    *,
    run_id: str,
    transitions: int,
    schema_id: str,
    exact_history: bool = True,
) -> float:
    state = replay.snapshot()
    rows = state.pop("rows")
    meta = {
        **state,
        "run_id": run_id,
        "transitions": transitions,
        "schema_id": schema_id,
        "format_version": 1,
        "exact_history": exact_history,
    }
    return atomic_write(
        path,
        lambda file: np.savez_compressed(
            file, rows=rows, meta=np.array(json.dumps(meta))
        ),
    )


def load_replay(
    path: Path, checkpoint: dict[str, Any], schema_id: str
) -> tuple[ReplayBuffer, bool]:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"].item()))
        if (
            meta["format_version"] != 1
            or meta["schema_id"] != schema_id
            or meta["run_id"] != checkpoint["run_id"]
        ):
            raise ValueError("replay identity/schema mismatch")
        count = meta["transitions"]
        if type(count) is not int or not 0 <= count <= checkpoint["transitions"]:
            raise ValueError("replay is newer than checkpoint or has invalid counter")
        replay = ReplayBuffer.from_snapshot({**meta, "rows": data["rows"].copy()})
        rows = replay.snapshot()["rows"]
        if meta["exact_history"] and len(replay) != min(count, replay.capacity):
            raise ValueError("replay length inconsistent with counter")
        if meta["exact_history"] and meta["write"] != count % replay.capacity:
            raise ValueError("replay write position inconsistent with counter")
        ordered = (
            rows
            if len(replay) < replay.capacity
            else np.concatenate((rows[meta["write"] :], rows[: meta["write"]]))
        )
        if (
            len(ordered) > 1
            and (ordered["transition_id"][1:] <= ordered["transition_id"][:-1]).any()
        ):
            raise ValueError("replay chronological order is invalid")
        expected = np.arange(count - len(replay), count, dtype=np.uint64)
        if meta["exact_history"] and not np.array_equal(
            np.sort(rows["transition_id"]), expected
        ):
            raise ValueError("replay transition IDs inconsistent with counter")
    if (
        len(np.unique(rows["transition_id"])) != len(replay)
        or (rows["transition_id"] >= count).any()
    ):
        raise ValueError("invalid replay transition IDs")
    return replay, count < checkpoint["transitions"] or not meta["exact_history"]


def save_checkpoint(path: Path, state: dict[str, Any]) -> float:
    return atomic_write(path, lambda file: cast(Any, torch).save(state, file))


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        state = cast(Any, torch).load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, EOFError, IndexError, RuntimeError) as error:
        raise ValueError(f"invalid checkpoint: {path}") from error
    if (
        not isinstance(state, dict)
        or cast(dict[str, Any], state).get("format_version") != 1
    ):
        raise ValueError("invalid checkpoint format")
    return cast(dict[str, Any], state)
