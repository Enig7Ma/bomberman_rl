"""Transition-budget driver saving the agent's full state at chunk boundaries.

Automatic per-round saves are disabled while managed by this driver. A killed
chunk is replayed from its last full checkpoint; uncommitted metrics are removed.
The checkpoint contains the small driver cursor and committed chunk records, so
a crash before writing chunks.jsonl does not accidentally train a chunk twice.
Files remain individually atomic, not a multi-file transaction: incompatible
checkpoint/replay generations fail closed through the agent's loader.
"""

import json
import logging
import random
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from tqdm import tqdm

from agent_code.dqn_agent.callbacks import AgentSelf, setup
from agent_code.dqn_agent.config import Config
from agent_code.dqn_agent.encoder import ENCODERS
from agent_code.dqn_agent.learner import Learner
from agent_code.dqn_agent.persistence import load_checkpoint, load_replay
from agent_code.dqn_agent.train import Trainer
from tournament.engine import quiet_logging, reset_framework_logging
from training.config import FROZEN, Curriculum, Stage
from training.driver import (
    CHUNKS_FILE,
    METRICS_FILE,
    SNAPSHOT_DIR,
    ChunkRecord,
    copy_atomic,
    environment,
    snapshots,
    world_seed,
)
from training.frozen import frozen_name, materialise_agent
from training.spec import SPECS
from training.world import create_training_world, play_round


def stage_config(course: Curriculum, stage: Stage, index: int, seed: int) -> Config:
    params: dict[str, Any] = {
        **course.params,
        **stage.params,
        "seed": seed,
        "init_seed": seed,
        "stage": index,
        "stage_transitions": stage.transitions,
        "epsilon_start": stage.epsilon_start,
        "epsilon_end": stage.epsilon_end,
        "epsilon_fraction": stage.decay_share,
    }
    return Config(**params)


def _publish(run_dir: Path, state: dict[str, Any]) -> None:
    """Reconstruct derived files only after the full-save hook succeeded."""
    rows = state["records"]
    path = run_dir / CHUNKS_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    tmp.replace(path)
    if rows and rows[-1]["snapshot"]:
        snapshot = run_dir / rows[-1]["snapshot"]
        if not snapshot.exists():
            copy_atomic(run_dir / "q_net.npz", snapshot)


def _trim_metrics(run_dir: Path, rounds: int) -> None:
    path = run_dir / METRICS_FILE
    if not path.exists():
        return
    kept: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        # A process may have died during the append of its last uncommitted line.
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            break
        if record["rounds_trained"] <= rounds:
            kept.append(line)
    path.write_text("".join(line + "\n" for line in kept), encoding="utf-8")


def train_dqn(
    course: Curriculum, run_dir: Path, seed: int, *, progress: bool
) -> list[ChunkRecord]:
    spec = SPECS[course.agent]
    checkpoint = run_dir / "checkpoint.pt"
    state: dict[str, Any] = {"next_stage": 0, "stage_rounds": 0, "records": []}
    stored: dict[str, Any] | None = None
    if checkpoint.exists():
        stored = load_checkpoint(checkpoint)
        _, degraded = load_replay(
            run_dir / "replay.npz",
            stored,
            ENCODERS[Config(**course.params).encoder].schema_id,
        )
        if degraded or not stored["exact_history"]:
            raise RuntimeError(
                "driver requires a consistent full save; "
                "stale replay is NOT exact resume"
            )
        cursor = stored.get("driver_state")
        if cursor is None:
            raise ValueError(
                "checkpoint has no curriculum cursor; "
                "use --init-from q_net.npz for a new run"
            )
        state = cast(dict[str, Any], cursor)
        _trim_metrics(run_dir, stored["rounds_trained"])
    first = len(state["records"])
    # Re-export even on a finished run: q_net may be older if the process died
    # after checkpoint replacement. This does not touch learner or replay RNG.
    if stored is not None:
        holder = cast(
            AgentSelf,
            SimpleNamespace(train=True, logger=logging.getLogger("training.dqn")),
        )
        with environment(
            {
                f"{spec.prefix}_MODEL": str(run_dir / spec.model_file),
                f"{spec.prefix}_PARAMS": json.dumps(stored["config"]),
            }
        ):
            setup(holder)
            restored = Trainer(holder)
        net = restored.learner.export()
        net.meta.update(
            {
                key: stored[key]
                for key in ("transitions", "rounds_trained", "run_id", "exact_history")
            }
        )
        net.meta.update(
            stage=str(restored.config.stage),
            parent_checkpoint=restored.parent_checkpoint,
        )
        net.save(run_dir / spec.model_file)
        _publish(run_dir, state)

    bar = tqdm(disable=not progress, unit="chunk", desc=run_dir.name, initial=first)
    try:
        while state["next_stage"] < len(course.stages):
            index = len(state["records"])
            stage_index = int(state["next_stage"])
            stage = course.stages[stage_index]
            config = stage_config(
                course, stage, stage_index + int(state.get("stage_offset", 0)), seed
            )
            rng = random.Random(world_seed(seed, index))
            source = stage
            replay = stage_index > 0 and rng.random() < stage.replay_share
            if replay:
                source = course.stages[rng.randrange(stage_index)]
            lineup = rng.choices(
                source.lineups, weights=[x.weight for x in source.lineups]
            )[0]
            scenario = lineup.scenario or source.scenario
            opponents = lineup.opponents
            params = stored["config"] if stored is not None else asdict(config)
            env = {
                f"{spec.prefix}_PARAMS": json.dumps(params),
                f"{spec.prefix}_MODEL": str(run_dir / spec.model_file),
                f"{spec.prefix}_METRICS": str(run_dir / METRICS_FILE),
            }
            if FROZEN in opponents:
                name = frozen_name(seed, spec=spec, namespace=run_dir)
                found = snapshots(run_dir)
                initial = run_dir / "frozen_initial.npz"
                if not found and not initial.exists():
                    live = run_dir / spec.model_file
                    if live.exists():
                        copy_atomic(live, initial)
                    else:
                        Learner(ENCODERS[config.encoder], config).export().save(initial)
                model = found[-1][1] if found else initial
                frozen_model = materialise_agent(
                    name,
                    model,
                    spec=spec,
                    params=course.params,
                    seed=seed,
                )
                env[f"{name.upper()}_MODEL"] = str(frozen_model)
                env[f"{name.upper()}_PARAMS"] = json.dumps(
                    {**course.params, "policy": "learned", "seed": seed}
                )
                opponents = tuple(name if x == FROZEN else x for x in opponents)
            started = time.perf_counter()
            with environment(env), quiet_logging():
                try:
                    world = create_training_world(
                        (spec.name, *opponents),
                        scenario,
                        world_seed(seed, index),
                        log_dir=str(run_dir / "logs" / f"chunk_{index:06d}"),
                    )
                    agent = cast(Any, world).agents[0].backend.runner.fake_self
                    trainer = cast(Trainer, agent.trainer)
                    trainer.managed_saves = True
                    trainer.configure_stage(config)
                    if stored is None:
                        trainer.driver_state = state
                        spec.save_chunk(agent)  # durable start, including private RNG
                        _trim_metrics(run_dir, 0)
                    before, updates = trainer.transitions, trainer.learner.updates
                    rounds = steps = 0
                    finished = False
                    while rounds < course.chunk_rounds and not finished:
                        play_round(world)
                        rounds += 1
                        steps += int(world.step)
                        state["stage_rounds"] += 1
                        finished = trainer.stage_position >= cast(
                            int, stage.transitions
                        ) or (
                            stage.rounds > 0 and state["stage_rounds"] >= stage.rounds
                        )
                    total = trainer.transitions
                    previous = (
                        state["records"][-1]["total_transitions"]
                        if state["records"]
                        else trainer.stage_start
                    )
                    take_snapshot = (
                        finished
                        or (total - trainer.stage_start)
                        // course.eval_every_transitions
                        > max(0, previous - trainer.stage_start)
                        // course.eval_every_transitions
                    )
                    snapshot = (
                        f"{SNAPSHOT_DIR}/transition_{total:09d}.npz"
                        if take_snapshot
                        else None
                    )
                    elapsed = time.perf_counter() - started
                    record = ChunkRecord(
                        index,
                        stage.name,
                        source.name,
                        replay,
                        scenario,
                        opponents,
                        trainer.learner.epsilon(
                            trainer.stage_position, config.stage_transitions
                        ),
                        world_seed(seed, index),
                        rounds,
                        steps,
                        elapsed,
                        rounds / elapsed,
                        steps / elapsed,
                        trainer.rounds_trained,
                        0,
                        snapshot,
                        total - before,
                        total,
                        trainer.learner.updates - updates,
                        trainer.learner.updates,
                        (total - before) / elapsed,
                        (trainer.learner.updates - updates) / elapsed,
                    )
                    state["records"].append(asdict(record))
                    if finished:
                        state["next_stage"] += 1
                        state["stage_rounds"] = 0
                    trainer.driver_state = state
                    spec.save_chunk(agent)
                    _publish(run_dir, state)
                finally:
                    reset_framework_logging()
            stored = load_checkpoint(checkpoint)
            bar.update()
    finally:
        bar.close()
    result: list[ChunkRecord] = []
    for row in state["records"][first:]:
        fields: dict[str, Any] = {**row, "opponents": tuple(row["opponents"])}
        result.append(ChunkRecord(**fields))
    return result
