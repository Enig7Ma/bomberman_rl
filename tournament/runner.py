"""Executing a schedule, optionally across worker processes.

Each worker gets its own ``log_dir`` so the framework's ``game.log`` handlers do
not fight over one file. The per-agent logs the framework writes under
``agent_code/<name>/logs/`` are *not* isolatable -- that path is hardcoded
relative to the working directory -- so workers do interleave there. That is
acceptable because the tournament never reads those logs, and ``quiet_logging``
keeps anything from being written to them in the first place.

Results are always returned in schedule order rather than completion order, so a
run's output never depends on how the work happened to be distributed.
"""

import os
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

from tqdm import tqdm

from tournament.engine import (
    DEFAULT_LOG_DIR,
    WorldConfig,
    ensure_repo_cwd,
    quiet_logging,
    reset_framework_logging,
    run_round,
)
from tournament.results import RoundResult

WORKER_LOG_ROOT = Path(DEFAULT_LOG_DIR) / "tournament"

# Set once per worker process by ``_init_worker``.
_worker_log_dir: str = DEFAULT_LOG_DIR


def _init_worker() -> None:
    global _worker_log_dir
    ensure_repo_cwd()
    _worker_log_dir = str(WORKER_LOG_ROOT / f"worker-{os.getpid()}")
    Path(_worker_log_dir).mkdir(parents=True, exist_ok=True)
    reset_framework_logging()


def _run_one(config: WorldConfig) -> RoundResult:
    """Play one round in a worker process. Must stay importable for pickling."""
    with quiet_logging():
        return run_round(config, log_dir=_worker_log_dir)


def _run_sequential(
    schedule: Sequence[WorldConfig], log_dir: str
) -> Iterator[RoundResult]:
    ensure_repo_cwd()
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    with quiet_logging():
        for config in schedule:
            yield run_round(config, log_dir=log_dir)


def _run_parallel(schedule: Sequence[WorldConfig], jobs: int) -> Iterator[RoundResult]:
    # Imported lazily: a sequential run should not pay to import
    # multiprocessing, and spawning re-imports this module in each worker.
    import multiprocessing as mp

    WORKER_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    with context.Pool(processes=jobs, initializer=_init_worker) as pool:
        # imap yields in submission order regardless of completion order.
        yield from pool.imap(_run_one, schedule)


def run_schedule(
    schedule: Sequence[WorldConfig],
    *,
    jobs: int = 1,
    log_dir: str = DEFAULT_LOG_DIR,
    progress: bool = False,
) -> list[RoundResult]:
    """Play every round in ``schedule`` and return the results in that order.

    ``jobs=1`` runs in this process, which keeps tracebacks and debuggers
    usable; anything higher spawns a pool.
    """
    if jobs < 1:
        raise ValueError(f"jobs must be at least 1, got {jobs}")
    if not schedule:
        return []

    stream: Iterable[RoundResult] = (
        _run_sequential(schedule, log_dir)
        if jobs == 1
        else _run_parallel(schedule, jobs)
    )
    if progress:
        stream = tqdm(stream, total=len(schedule), unit="round")
    return list(stream)
