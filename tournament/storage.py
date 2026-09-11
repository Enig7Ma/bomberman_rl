"""Reading and writing round results as JSON Lines.

One round per line, appended as games finish. Keeping raw per-round records
rather than only aggregates means re-analysis -- a different statistic, a
different subset of lineups -- never has to replay the games.
"""

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from tournament.results import AgentRoundResult, RoundResult


def _agent_from_mapping(raw: Mapping[str, Any]) -> AgentRoundResult:
    # Driven by the dataclass fields so the reader keeps up with the schema;
    # a missing key raises rather than silently defaulting.
    return AgentRoundResult(**{f.name: raw[f.name] for f in fields(AgentRoundResult)})


def _round_from_mapping(raw: Mapping[str, Any]) -> RoundResult:
    agents: Iterable[Mapping[str, Any]] = raw["agents"]
    return RoundResult(
        scenario=raw["scenario"],
        seed=raw["seed"],
        lineup=tuple(raw["lineup"]),
        arm=raw["arm"],
        focus_seat=raw["focus_seat"],
        steps=raw["steps"],
        agents=tuple(_agent_from_mapping(agent) for agent in agents),
    )


def write_rounds(
    path: Path | str, results: Iterable[RoundResult], *, append: bool = False
) -> None:
    """Write results as JSON Lines, creating parent directories as needed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a" if append else "w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), sort_keys=True))
            handle.write("\n")


def read_rounds(path: Path | str) -> list[RoundResult]:
    """Read every round from a JSON Lines file, skipping blank lines."""
    with Path(path).open(encoding="utf-8") as handle:
        return [
            _round_from_mapping(json.loads(line)) for line in handle if line.strip()
        ]
