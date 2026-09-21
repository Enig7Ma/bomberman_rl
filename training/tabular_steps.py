"""Opt-in transition budgets for tables; legacy round driver is unchanged.

The atomic Q-table contains weights, visits, private RNG and committed cursor.
Metrics/chunk files are derived at chunk boundaries. Interrupted chunks replay
from the last committed table; stock opponents are still entropy-seeded.
"""

import json
import random
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from functools import wraps
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from agent_code.tabular_q_agent import callbacks
from agent_code.tabular_q_agent.config import Config
from agent_code.tabular_q_agent.features import ENCODINGS
from agent_code.tabular_q_agent.qtable import QTable
from tournament.engine import quiet_logging, reset_framework_logging
from training.config import FROZEN, Curriculum, Stage
from training.driver import (
    CHUNKS_FILE,
    METRICS_FILE,
    ChunkRecord,
    copy_atomic,
    environment,
    snapshots,
    world_seed,
)
from training.frozen import frozen_name, materialise_frozen
from training.world import create_training_world, play_round


def restore_rng(raw: list[Any]) -> tuple[Any, ...]:
    """JSON's lists back to Random's documented state tuple."""
    return tuple(
        restore_rng(cast(list[Any], v)) if isinstance(v, list) else v for v in raw
    )


def replace_derived(source: Path, target: Path) -> None:
    """Bounded retry for Windows sharing locks; preserve both files on failure."""
    for attempt in range(5):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


def publish(run_dir: Path, table: QTable) -> None:
    state = cast(dict[str, Any], table.meta["transition_driver"])
    metrics = run_dir / METRICS_FILE
    kept: list[str] = []
    if metrics.exists():
        for line in metrics.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            if row["rounds_trained"] > table.meta["rounds_trained"]:
                break
            kept.append(json.dumps(row) + "\n")
        tmp = metrics.with_suffix(".tmp")
        tmp.write_text("".join(kept), encoding="utf-8")
        replace_derived(tmp, metrics)
    tmp = run_dir / "chunks.tmp"
    tmp.write_text(
        "".join(json.dumps(r) + "\n" for r in state["records"]), encoding="utf-8"
    )
    replace_derived(tmp, run_dir / CHUNKS_FILE)
    if state["records"] and state["records"][-1]["snapshot"]:
        target = run_dir / state["records"][-1]["snapshot"]
        if not target.exists():
            copy_atomic(run_dir / "q_table.npz", target)


def train_tabular_steps(
    course: Curriculum, run_dir: Path, seed: int
) -> list[ChunkRecord]:
    model = run_dir / "q_table.npz"
    encoding = ENCODINGS[Config(**course.params).encoding]
    table = QTable.load(model, encoding) if model.exists() else QTable.zeros(encoding)
    total = int(table.meta.get("steps_trained", 0))
    if "transition_driver" not in table.meta:
        table.meta.setdefault("rounds_trained", 0)
        table.meta.setdefault("steps_trained", 0)
        table.meta["transition_driver"] = {
            "format_version": 1,
            "seed": seed,
            "origin": total,
            "stage_start": total,
            "stage_rounds": 0,
            "next_stage": 0,
            "records": [],
            "rng": random.Random(seed).getstate(),
        }
        table.save(model)
    state = cast(dict[str, Any], table.meta["transition_driver"])
    if state["format_version"] != 1 or state["seed"] != seed:
        raise ValueError("incompatible tabular transition cursor")
    publish(run_dir, table)
    first = len(state["records"])
    while state["next_stage"] < len(course.stages):
        index = len(state["records"])
        stage_index = int(state["next_stage"])
        stage = course.stages[stage_index]
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
        params = {
            **course.params,
            **stage.params,
            "seed": seed,
            "stage": stage.name,
            "epsilon": stage.epsilon_at_transition(total - state["stage_start"]),
            # Managed chunk save only, never a partially committed round save.
            "save_every": 2**63 - 1,
        }
        if FROZEN in opponents:
            name = frozen_name(seed, namespace=run_dir)
            found = snapshots(run_dir)
            materialise_frozen(
                name, found[-1][1] if found else model, encoding=encoding.name
            )
            opponents = tuple(name if x == FROZEN else x for x in opponents)
        original_act = callbacks.act
        counter = [total]

        @wraps(original_act)
        def act(
            agent: callbacks.AgentSelf,
            observation: callbacks.GameState,
            counter: list[int] = counter,
            start: int = state["stage_start"],
            stage: Stage = stage,
            original_act: Callable[
                [callbacks.AgentSelf, callbacks.GameState], callbacks.Action
            ] = original_act,
        ) -> callbacks.Action:
            if agent.train:
                epsilon = stage.epsilon_at_transition(counter[0] - start)
                agent.config = replace(agent.config, epsilon=epsilon)
                if agent.trainer is not None:
                    agent.trainer.epsilon = epsilon
            result = original_act(agent, observation)
            if agent.train:
                counter[0] += 1
            return result

        before = total
        rounds = steps = 0
        started = time.perf_counter()
        with (
            environment(
                {
                    "TABULAR_Q_AGENT_MODEL": str(model),
                    "TABULAR_Q_AGENT_METRICS": str(run_dir / METRICS_FILE),
                    "TABULAR_Q_AGENT_PARAMS": json.dumps(params),
                }
            ),
            quiet_logging(),
            patch.object(callbacks, "act", act),
        ):
            try:
                world = create_training_world(
                    ("tabular_q_agent", *opponents),
                    scenario,
                    world_seed(seed, index),
                    log_dir=str(run_dir / "logs" / f"chunk_{index:06d}"),
                )
                agent = cast(Any, world).agents[0].backend.runner.fake_self
                agent.rng.setstate(restore_rng(list(state["rng"])))
                finished = False
                while rounds < course.chunk_rounds and not finished:
                    play_round(world)
                    rounds += 1
                    steps += int(world.step)
                    state["stage_rounds"] += 1
                    total = int(agent.table.meta["steps_trained"])
                    assert total == counter[0]
                    finished = total - state["stage_start"] >= cast(
                        int, stage.transitions
                    ) or (stage.rounds > 0 and state["stage_rounds"] >= stage.rounds)
                position = total - state["stage_start"]
                snapshot = (
                    f"snapshots/transition_{total:09d}.npz"
                    if finished
                    or position // course.eval_every_transitions
                    > (before - state["stage_start"]) // course.eval_every_transitions
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
                    stage.epsilon_at_transition(position),
                    world_seed(seed, index),
                    rounds,
                    steps,
                    elapsed,
                    rounds / elapsed,
                    steps / elapsed,
                    agent.trainer.rounds_trained,
                    agent.table.visited_states,
                    snapshot,
                    total - before,
                    total,
                    total - before,
                    total,
                    (total - before) / elapsed,
                    (total - before) / elapsed,
                )
                state["records"].append(asdict(record))
                state["rng"] = agent.rng.getstate()
                if finished:
                    state["next_stage"] += 1
                    state["stage_start"] = total
                    state["stage_rounds"] = 0
                agent.table.meta["config"] = asdict(agent.config)
                agent.table.meta["transition_driver"] = state
                agent.table.save(model)
                table = agent.table
                publish(run_dir, table)
            finally:
                reset_framework_logging()
    result: list[ChunkRecord] = []
    for r in state["records"][first:]:
        fields: dict[str, Any] = {**r, "opponents": tuple(r["opponents"])}
        result.append(ChunkRecord(**fields))
    return result
