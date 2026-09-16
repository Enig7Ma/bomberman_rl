"""Averaging a run's last snapshots into one table.

Found in Q7 (``dev/experiments/tabular.md``): greedy play from a single
snapshot flips between productive and collapsed tables, because in common
states the action values differ by less than the update noise. The mean of the
last few snapshots keeps each action's level and averages the noise away, like
Polyak averaging of network weights. It is a post-training step: training
itself continues from the live table, never from an average.

The averaged table takes its visit counts and metadata from the newest
snapshot, adds ``meta["averaged_from"]``, and is written to
``averaged/last<K>_round_<N>.npz`` in the run directory, where ``N`` is the
newest snapshot's round.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np

from agent_code.tabular_q_agent.features import ENCODINGS, Encoding
from agent_code.tabular_q_agent.qtable import QTable
from training.driver import load_run, snapshots

AVERAGED_DIR: Final = "averaged"


def average_tables(paths: Sequence[Path], encoding: Encoding) -> QTable:
    """The mean Q-values of ``paths``; counts and metadata of the last one."""
    if not paths:
        raise ValueError("need at least one table to average")
    tables = [QTable.load(path, encoding) for path in paths]
    result = tables[-1]
    result.q = np.stack([table.q for table in tables]).mean(axis=0).astype(np.float32)
    result.meta = {**result.meta, "averaged_from": [str(path) for path in paths]}
    return result


def average_run(run_dir: Path, last: int) -> Path:
    """Average the run's ``last`` newest snapshots; returns the written table."""
    if last < 1:
        raise ValueError(f"last must be at least 1, got {last}")
    curriculum, _ = load_run(run_dir)
    encoding = ENCODINGS[curriculum.agent_config().encoding]
    found = snapshots(run_dir)
    if len(found) < last:
        raise ValueError(f"{run_dir} has {len(found)} snapshots, fewer than {last}")
    chosen = found[-last:]
    newest = chosen[-1][0]
    out = run_dir / AVERAGED_DIR / f"last{last}_round_{newest:06d}.npz"
    average_tables([path for _, path in chosen], encoding).save(out)
    return out
