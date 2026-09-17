"""Append-only JSONL training diagnostics, without Torch dependencies."""

import json
from pathlib import Path
from typing import Any


def append_record(path: Path, record: dict[str, Any]) -> None:
    text = json.dumps(record, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(text + "\n")


def read_records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
