"""Per-round training records, appended as one JSON object per line.

Records are append-only so a killed run keeps everything up to its last
finished round, and a learning curve can be re-analysed without replaying any
games. ``base_reward`` is the unshaped score and is what model selection uses;
``shaped_return`` includes aids and shaping and is for debugging only.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RoundRecord:
    # Round number in the current world, and rounds this table has trained.
    round: int
    rounds_trained: int
    stage: str
    # Names of the other agents alive at the round's first step.
    opponents: list[str]
    # Actions taken, and Q-updates applied (one per action).
    steps: int
    updates: int
    survived: bool
    died: bool
    self_kill: bool
    base_reward: float
    shaped_return: float
    coins: int
    kills: int
    crates: int
    bombs: int
    invalid: int
    # Every engine event this agent received, counted once each.
    event_counts: dict[str, int]
    epsilon: float
    mean_alpha: float
    mean_abs_td: float
    # States with any updated action, after this round and gained in it.
    visited_states: int
    new_states: int
    # Share of steps where the mask left exactly one action.
    forced_fraction: float
    explored_steps: int
    # Greedy decisions in a state whose allowed actions were all unvisited.
    unseen_decisions: int
    wall_time: float


def append_record(path: Path, record: RoundRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(asdict(record)) + "\n")


def read_records(path: Path) -> list[RoundRecord]:
    records: list[RoundRecord] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                fields: dict[str, Any] = json.loads(line)
                records.append(RoundRecord(**fields))
    return records
