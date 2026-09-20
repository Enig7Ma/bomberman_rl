"""Short, one-process measurements; use a NEW output directory each time.

python -m docs.experiments.dqn_d5_benchmark --out results/dqn/d5_throughput

End-to-end time includes interpreter/Torch startup, warm-up, all saves and
snapshots. Round time (metrics) excludes setup and final chunk saves, but includes
warm-up. These are plumbing measurements, not learned-performance comparisons.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from tournament.engine import REPO_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out: Path = args.out.resolve()
    if out.exists():
        parser.error("--out must be new, otherwise a resume would distort timing")
    out.mkdir(parents=True)
    summary: dict[str, Any] = {}
    for agent in ("dqn_agent", "tabular_q_agent"):
        config = REPO_ROOT / "docs" / "experiments" / "dqn_d5" / f"{agent}.json"
        command = [
            sys.executable,
            "-m",
            "training",
            "run",
            "--curriculum",
            str(config),
            "--out",
            str(out / agent),
            "--runs",
            "1",
            "--jobs",
            "1",
            "--no-progress",
        ]
        started = time.perf_counter()
        with (out / f"{agent}.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        elapsed = time.perf_counter() - started
        path = out / agent / "run_0" / "metrics.jsonl"
        records: list[dict[str, Any]] = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]
        transitions = sum(row["steps"] for row in records)
        updates = (
            records[-1]["updates"]
            if agent == "dqn_agent"
            else sum(row["updates"] for row in records)
        )
        round_seconds = sum(row["wall_time"] for row in records)
        summary[agent] = {
            "command": command,
            "rounds": len(records),
            "transitions": transitions,
            "updates": updates,
            "end_to_end_seconds": elapsed,
            "transitions_per_second": transitions / elapsed,
            "rounds_per_second": len(records) / elapsed,
            "updates_per_second": updates / elapsed,
            "round_seconds": round_seconds,
            "round_only_transitions_per_second": transitions / round_seconds,
            "round_only_rounds_per_second": len(records) / round_seconds,
            "round_only_updates_per_second": updates / round_seconds,
            "warmup_included": True,
            "startup_and_saving_in_end_to_end": True,
        }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
