"""Running a curriculum (plan ``dev/tabiular_q-learning.md`` step Q5).

A run trains one Q-table through the stages of a ``Curriculum``:

- **Chunks.** Rounds are played in chunks of ``chunk_rounds``, each in a fresh
  ``TrainingWorld`` with its own seed. Lineup, scenario and epsilon are fixed
  within a chunk and chosen per chunk: a stage's lineups are sampled by weight,
  a ``replay_share`` of chunks replays an earlier stage (against forgetting),
  and epsilon follows the stage's linear schedule.
- **Checkpoints.** The agent saves its table at the end of every chunk
  (``save_every = chunk_rounds``), and after each chunk the driver checks that
  the saved table advanced by exactly one chunk. A killed run resumes where it
  stopped: ``train_run`` on the same directory skips the chunks the table has
  already trained.
- **Snapshots.** Every ``eval_every`` rounds the table is copied to
  ``snapshots/round_NNNNNN.npz`` for ``training.evaluate``, and the ``frozen``
  opponent is switched to that snapshot.
- **Continuing a table.** ``init_from`` starts a run from another run's table:
  it is copied into the run directory, its ``rounds_trained`` carries on, and
  the chunk plan starts at the curriculum's first chunk. ``run.json`` records
  the source table, which is never written to.
- **Seeds.** World seeds come from a range no evaluation uses:
  ``100000 + 10000 * run_seed + chunk index``. The chunk plan depends on
  ``run_seed`` alone. Opponents seed themselves from entropy, so two runs with
  one seed still play different games.

The agent is configured only through its environment variables, as the
tournament harness does, so the code that trains is the code that plays.

A run directory holds ``run.json`` (seed, curriculum, git commit),
``q_table.npz``, ``metrics.jsonl`` (one record per round, written by the
agent), ``chunks.jsonl`` (one record per chunk, with throughput),
``snapshots/`` and ``logs/``.
"""

import json
import os
import random
import re
import shutil
import subprocess
import time
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from tqdm import tqdm

from agent_code.tabular_q_agent.config import ENV_VAR, METRICS_ENV_VAR, MODEL_ENV_VAR
from agent_code.tabular_q_agent.features import ENCODINGS
from agent_code.tabular_q_agent.qtable import QTable
from tournament.engine import REPO_ROOT, quiet_logging, reset_framework_logging
from training.config import FROZEN, LEARNER, Curriculum, parse_curriculum
from training.frozen import env_prefix, frozen_name, install_table, materialise_frozen
from training.world import create_training_world, play_round

TRAINING_SEED_BASE: Final = 100_000
SEEDS_PER_RUN: Final = 10_000

RUN_FILE: Final = "run.json"
MODEL_FILE: Final = "q_table.npz"
METRICS_FILE: Final = "metrics.jsonl"
CHUNKS_FILE: Final = "chunks.jsonl"
SNAPSHOT_DIR: Final = "snapshots"
_SNAPSHOT = re.compile(r"round_(\d+)\.npz")


@dataclass(frozen=True)
class Chunk:
    index: int
    stage: str
    # The stage whose scenario and lineups this chunk plays: ``stage`` itself,
    # or an earlier one when ``replay``.
    source: str
    replay: bool
    scenario: str
    opponents: tuple[str, ...]
    epsilon: float
    seed: int

    @property
    def label(self) -> str:
        return f"{self.stage}/replay:{self.source}" if self.replay else self.stage


@dataclass(frozen=True)
class ChunkRecord:
    index: int
    stage: str
    source: str
    replay: bool
    scenario: str
    opponents: tuple[str, ...]
    epsilon: float
    seed: int
    rounds: int
    engine_steps: int
    wall_time: float
    rounds_per_second: float
    steps_per_second: float
    rounds_trained: int
    visited_states: int
    snapshot: str | None


def world_seed(run_seed: int, index: int) -> int:
    if run_seed < 0 or not 0 <= index < SEEDS_PER_RUN:
        raise ValueError(f"no training seed for run {run_seed}, chunk {index}")
    return TRAINING_SEED_BASE + SEEDS_PER_RUN * run_seed + index


def plan_chunks(curriculum: Curriculum, run_seed: int) -> list[Chunk]:
    """Every chunk of the run, in order; a function of ``run_seed`` only."""
    rng = random.Random(run_seed)
    chunks: list[Chunk] = []
    for stage_index, stage in enumerate(curriculum.stages):
        for done in range(0, stage.rounds, curriculum.chunk_rounds):
            source = stage
            replay = stage_index > 0 and rng.random() < stage.replay_share
            if replay:
                source = curriculum.stages[rng.randrange(stage_index)]
            weights = [lineup.weight for lineup in source.lineups]
            lineup = rng.choices(source.lineups, weights=weights)[0]
            index = len(chunks)
            chunks.append(
                Chunk(
                    index=index,
                    stage=stage.name,
                    source=source.name,
                    replay=replay,
                    scenario=lineup.scenario or source.scenario,
                    opponents=lineup.opponents,
                    epsilon=stage.epsilon_at(done),
                    seed=world_seed(run_seed, index),
                )
            )
    return chunks


@contextmanager
def environment(values: Mapping[str, str]) -> Generator[None]:
    """Set environment variables for the block, then restore the old ones."""
    saved = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, old in saved.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old


def snapshots(run_dir: Path) -> list[tuple[int, Path]]:
    """``(rounds trained, path)`` of every snapshot, oldest first."""
    found: list[tuple[int, Path]] = []
    directory = run_dir / SNAPSHOT_DIR
    if directory.is_dir():
        for path in directory.iterdir():
            match = _SNAPSHOT.fullmatch(path.name)
            if match:
                found.append((int(match.group(1)), path))
    return sorted(found)


def load_run(run_dir: Path) -> tuple[Curriculum, int]:
    """The curriculum and run seed a run directory was started with."""
    stored: Any = json.loads((run_dir / RUN_FILE).read_text(encoding="utf-8"))
    return parse_curriculum(stored["curriculum"]), int(stored["run_seed"])


def _git_commit() -> str | None:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{head}-dirty" if dirty else head


def _start_or_resume(
    run_dir: Path, curriculum: Curriculum, run_seed: int, init_from: Path | None
) -> int:
    """Create or check ``run.json``; returns the rounds the start table had."""
    run_file = run_dir / RUN_FILE
    wanted_init = str(init_from.resolve()) if init_from is not None else None
    if run_file.exists():
        stored: Any = json.loads(run_file.read_text(encoding="utf-8"))
        if stored["run_seed"] != run_seed:
            raise RuntimeError(
                f"{run_dir} is run seed {stored['run_seed']}, not {run_seed}"
            )
        # Compared after parsing, so fields added later with defaults still match.
        if asdict(parse_curriculum(stored["curriculum"])) != asdict(curriculum):
            raise RuntimeError(f"{run_dir} was started with a different curriculum")
        if wanted_init is not None and stored.get("init_from") != wanted_init:
            started_from = stored.get("init_from")
            raise RuntimeError(
                f"{run_dir} was started from {started_from}, not {wanted_init}"
            )
        return int(stored.get("init_rounds", 0))

    run_dir.mkdir(parents=True, exist_ok=True)
    init_rounds = 0
    if init_from is not None:
        encoding = ENCODINGS[curriculum.agent_config().encoding]
        source = QTable.load(init_from, encoding)  # refuses another encoding
        init_rounds = int(source.meta.get("rounds_trained", 0))
        _copy_atomic(init_from, run_dir / MODEL_FILE)
    document = {
        "run_seed": run_seed,
        "git_commit": _git_commit(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "init_from": wanted_init,
        "init_rounds": init_rounds,
        "curriculum": asdict(curriculum),
    }
    run_file.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return init_rounds


def _rounds_trained(model: Path, curriculum: Curriculum) -> int:
    if not model.exists():
        return 0
    encoding = ENCODINGS[curriculum.agent_config().encoding]
    return int(QTable.load(model, encoding).meta.get("rounds_trained", 0))


def _copy_atomic(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.tmp")
    shutil.copyfile(source, tmp)
    os.replace(tmp, target)


def train_run(
    curriculum: Curriculum,
    run_dir: Path,
    run_seed: int,
    *,
    init_from: Path | None = None,
    progress: bool = False,
) -> list[ChunkRecord]:
    """Play every chunk the table has not trained yet; returns their records.

    ``init_from`` (only read when the run directory is new) starts from a copy
    of another table instead of an empty one.
    """
    run_dir = run_dir.resolve()
    init_rounds = _start_or_resume(run_dir, curriculum, run_seed, init_from)
    model = run_dir / MODEL_FILE
    encoding = curriculum.agent_config().encoding
    chunks = plan_chunks(curriculum, run_seed)

    done = _rounds_trained(model, curriculum) - init_rounds
    if done < 0 or done % curriculum.chunk_rounds:
        raise RuntimeError(
            f"{model} has trained {done} rounds in this run, not a multiple of the "
            f"chunk size {curriculum.chunk_rounds}"
        )
    remaining = chunks[done // curriculum.chunk_rounds :]

    frozen: str | None = None
    if any(FROZEN in chunk.opponents for chunk in remaining):
        frozen = frozen_name(run_seed)
        latest = snapshots(run_dir)
        start_table = latest[-1][1] if latest else (model if model.exists() else None)
        materialise_frozen(frozen, start_table, encoding=encoding)

    records: list[ChunkRecord] = []
    iterator: Sequence[Chunk] | tqdm[Chunk] = remaining
    if progress:
        iterator = tqdm(remaining, unit="chunk", desc=run_dir.name)
    for chunk in iterator:
        records.append(_play_chunk(curriculum, run_dir, chunk, frozen))
    return records


def _play_chunk(
    curriculum: Curriculum, run_dir: Path, chunk: Chunk, frozen: str | None
) -> ChunkRecord:
    model = run_dir / MODEL_FILE
    encoding = curriculum.agent_config().encoding
    params = {
        **curriculum.params,
        "epsilon": chunk.epsilon,
        "stage": chunk.label,
        "save_every": curriculum.chunk_rounds,
        "seed": chunk.seed,
    }
    env = {
        MODEL_ENV_VAR: str(model),
        METRICS_ENV_VAR: str(run_dir / METRICS_FILE),
        ENV_VAR: json.dumps(params),
    }
    opponents = tuple(
        frozen if name == FROZEN and frozen is not None else name
        for name in chunk.opponents
    )
    if frozen is not None and frozen in opponents:
        env[f"{env_prefix(frozen)}_PARAMS"] = json.dumps(curriculum.params)

    before = _rounds_trained(model, curriculum)
    engine_steps = 0
    started = time.perf_counter()
    with environment(env), quiet_logging():
        try:
            world = create_training_world(
                (LEARNER, *opponents),
                chunk.scenario,
                chunk.seed,
                log_dir=str(run_dir / "logs"),
            )
            for _ in range(curriculum.chunk_rounds):
                play_round(world)
                engine_steps += int(world.step)
        finally:
            reset_framework_logging()
    wall = time.perf_counter() - started

    table = QTable.load(model, ENCODINGS[encoding])
    trained = int(table.meta["rounds_trained"])
    if trained != before + curriculum.chunk_rounds:
        raise RuntimeError(
            f"chunk {chunk.index}: the saved table has {trained} rounds, expected "
            f"{before + curriculum.chunk_rounds}"
        )

    snapshot: Path | None = None
    if trained % curriculum.eval_every == 0:
        snapshot = run_dir / SNAPSHOT_DIR / f"round_{trained:06d}.npz"
        _copy_atomic(model, snapshot)
        if frozen is not None:
            install_table(frozen, snapshot, encoding=encoding)

    record = ChunkRecord(
        index=chunk.index,
        stage=chunk.stage,
        source=chunk.source,
        replay=chunk.replay,
        scenario=chunk.scenario,
        opponents=opponents,
        epsilon=chunk.epsilon,
        seed=chunk.seed,
        rounds=curriculum.chunk_rounds,
        engine_steps=engine_steps,
        wall_time=wall,
        rounds_per_second=curriculum.chunk_rounds / wall,
        steps_per_second=engine_steps / wall,
        rounds_trained=trained,
        visited_states=table.visited_states,
        snapshot=snapshot.name if snapshot is not None else None,
    )
    with (run_dir / CHUNKS_FILE).open("a", encoding="utf-8") as file:
        file.write(json.dumps(asdict(record)) + "\n")
    return record


def read_chunk_records(run_dir: Path) -> list[ChunkRecord]:
    path = run_dir / CHUNKS_FILE
    if not path.exists():
        return []
    records: list[ChunkRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            raw: dict[str, Any] = json.loads(line)
            raw["opponents"] = tuple(raw["opponents"])
            records.append(ChunkRecord(**raw))
    return records


def _train_worker(
    curriculum: Curriculum, run_dir: Path, run_seed: int, init_from: Path | None
) -> None:
    train_run(curriculum, run_dir, run_seed, init_from=init_from)


def run_many(
    curriculum: Curriculum,
    out_dir: Path,
    run_seeds: Sequence[int],
    *,
    jobs: int = 1,
    init_from: Path | None = None,
    progress: bool = False,
) -> dict[int, Path]:
    """Independent runs ``out_dir/run_<seed>``, ``jobs`` at a time."""
    if jobs < 1:
        raise ValueError(f"jobs must be at least 1, got {jobs}")
    run_dirs = {seed: (out_dir / f"run_{seed}").resolve() for seed in run_seeds}
    if jobs == 1 or len(run_dirs) == 1:
        for seed, run_dir in run_dirs.items():
            train_run(curriculum, run_dir, seed, init_from=init_from, progress=progress)
        return run_dirs
    import multiprocessing as mp

    context = mp.get_context("spawn")
    work = [
        (curriculum, run_dir, seed, init_from) for seed, run_dir in run_dirs.items()
    ]
    with context.Pool(processes=min(jobs, len(work))) as pool:
        pool.starmap(_train_worker, work)
    return run_dirs
