"""Measure the wall time of one CPU learner update, with machine provenance.

Writes a single JSON report and refuses to overwrite it. Run from the
repository root::

    uv run python -m docs.experiments.dqn_bench_update \\
        --output results/dqn/bench_update.json
"""

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch

from agent_code.dqn_agent.config import Config
from agent_code.dqn_agent.encoder import OneHotE3
from agent_code.dqn_agent.learner import Learner
from agent_code.dqn_agent.replay import ReplayBuffer, ReplayTransition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    learner = Learner(OneHotE3(), Config(init_seed=0, seed=0))
    replay = ReplayBuffer(64, 32)
    rng = np.random.default_rng(0)
    for i in range(64):
        replay.push(
            ReplayTransition(
                x=rng.normal(size=32).astype(np.float32),
                a=i % 6,
                base=float(rng.normal()),
                crates=0,
                deaths=0,
                phi_unit=0,
                phi_unit_next=0,
                x_next=rng.normal(size=32).astype(np.float32),
                mask_next=np.array([True, True, False, True, False, False]),
                done=i % 4 == 0,
                stage=0,
                round_id=0,
                transition_id=i,
            )
        )
    batch = learner.sample(replay)
    for _ in range(200):
        learner.update(batch)
    durations: list[float] = []
    syncs = 0
    for _ in range(1000):
        start = perf_counter_ns()
        stats = learner.update(batch)
        durations.append((perf_counter_ns() - start) / 1e6)
        syncs += int(stats.target_synced)
    report = {
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "config": asdict(learner.config),
        "architecture": [32, 128, 128, 6],
        "cpu_threads": torch.get_num_threads(),
        "warmup_updates": 200,
        "measured_updates": 1000,
        "batch_size": 64,
        "target_syncs_measured": syncs,
        "scope": (
            "Learner.update incl conversion, validation, target, backward, "
            "Adam, clipping, stats and scheduled sync; "
            "excludes replay sampling and warmup"
        ),
        "mean_ms": float(np.mean(durations)),
        "median_ms": float(np.median(durations)),
        "p95_ms": float(np.percentile(durations, 95)),
        "max_ms": max(durations),
        "samples_ms": durations,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "samples_ms"}, indent=2))


if __name__ == "__main__":
    main()
