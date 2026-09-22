"""Learning curves: every snapshot of a run, played by the tournament harness.

Training records are taken while exploring, against whatever the curriculum
happened to sample, so they are not a measure of playing strength. A snapshot
is evaluated instead as the tournament would see it: the agent loads the
snapshot through ``TABULAR_Q_AGENT_MODEL``, plays greedily (``train=False``),
on a fixed preset and a fixed seed range that training never uses.

Results are stored per snapshot in ``eval/<preset>_round_NNNNNN.jsonl`` and
reused when present, so evaluating a run again only plays new snapshots. The
curve is written to ``eval/<preset>_curve.md``. Next to the score it shows
two numbers from training, averaged over the ``eval_every`` rounds before the
snapshot: unseen-state decisions per round and the share of steps where the
mask left a single action.
"""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tournament.runner import run_schedule
from tournament.schedule import CANDIDATE_ARM, PRESETS, build_schedule, seeds
from tournament.stats import ArmSummary, summarise_arm
from tournament.storage import read_rounds, write_rounds
from training.driver import (
    METRICS_FILE,
    environment,
    load_run,
    read_chunk_records,
    snapshots,
)
from training.spec import SPECS

EVAL_DIR = "eval"


@dataclass(frozen=True)
class CurvePoint:
    rounds_trained: int
    summary: ArmSummary
    visited_states: int
    # From training, over the ``eval_every`` rounds before the snapshot; NaN
    # when no training record falls in that window.
    unseen_per_round: float
    forced_fraction: float
    transitions: int | None = None


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def evaluate_run(
    run_dir: Path, *, jobs: int = 1, progress: bool = False
) -> dict[str, list[CurvePoint]]:
    """Evaluate every snapshot on every preset of the run's curriculum."""
    curriculum, _ = load_run(run_dir)
    spec = SPECS[curriculum.agent]
    dqn = curriculum.agent == "dqn_agent"
    by_transitions = curriculum.stages[0].transitions is not None
    metrics_file = run_dir / METRICS_FILE
    records: list[dict[str, Any]] = (
        [
            json.loads(line)
            for line in metrics_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if metrics_file.exists()
        else []
    )
    if by_transitions and not dqn:
        chunks = read_chunk_records(run_dir)
        total = chunks[0].total_transitions - chunks[0].transitions if chunks else 0
        for record in records:
            total += int(record["steps"])
            record["transitions"] = total

    curves: dict[str, list[CurvePoint]] = {}
    for evaluation in curriculum.evaluations:
        schedule = build_schedule(
            spec.name,
            PRESETS[evaluation.preset],
            seeds(evaluation.seeds, evaluation.seed_start),
            with_control=False,
        )
        points: list[CurvePoint] = []
        for position, snapshot in snapshots(run_dir):
            rounds_trained, visited = spec.model_info(snapshot, curriculum.params)
            unit = "transition" if by_transitions else "round"
            out = (
                run_dir / EVAL_DIR / f"{evaluation.preset}_{unit}_{position:06d}.jsonl"
            )
            if out.exists():
                results = read_rounds(out)
            else:
                env = {
                    f"{spec.prefix}_MODEL": str(snapshot.resolve()),
                    f"{spec.prefix}_PARAMS": json.dumps(
                        {
                            **curriculum.params,
                            **(
                                {"policy": "learned", "seed": evaluation.seed_start}
                                if dqn
                                else {}
                            ),
                        }
                    ),
                }
                with environment(env):
                    results = run_schedule(schedule, jobs=jobs, progress=progress)
                write_rounds(out, results)
            window = [
                record
                for record in records
                if position
                - (
                    curriculum.eval_every_transitions
                    if by_transitions
                    else curriculum.eval_every
                )
                < record["transitions" if by_transitions else "rounds_trained"]
                <= position
            ]
            points.append(
                CurvePoint(
                    rounds_trained=rounds_trained,
                    summary=summarise_arm(results, CANDIDATE_ARM),
                    visited_states=visited,
                    transitions=position if by_transitions else None,
                    unseen_per_round=_mean(
                        [float(r.get("unseen_decisions", math.nan)) for r in window]
                    ),
                    forced_fraction=_mean([r["forced_fraction"] for r in window]),
                )
            )
        title = f"{run_dir.name}: {evaluation.preset}, seeds {evaluation.seed_start}+"
        curve = render_curve(points, title)
        (run_dir / EVAL_DIR).mkdir(parents=True, exist_ok=True)
        (run_dir / EVAL_DIR / f"{evaluation.preset}_curve.md").write_text(
            curve, encoding="utf-8"
        )
        curves[evaluation.preset] = points
    return curves


def render_curve(points: Sequence[CurvePoint], title: str) -> str:
    axis = (
        "transitions"
        if points and points[0].transitions is not None
        else "rounds trained"
    )
    lines = [
        f"## {title}",
        "",
        f"| {axis} | rounds | score [95% CI] | win | survive | suicides "
        "| coins | kills | round len | visited states | unseen / round (train) "
        "| forced (train) |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point in points:
        s = point.summary
        score = f"{s.score.mean:.2f} [{s.score.ci.low:.2f}, {s.score.ci.high:.2f}]"
        position = (
            point.transitions if point.transitions is not None else point.rounds_trained
        )
        lines.append(
            f"| {position} | {s.rounds} | {score} | {s.win_rate:.2f} "
            f"| {s.survival_rate:.2f} | {s.suicides:.2f} | {s.coins:.2f} "
            f"| {s.kills:.2f} | {s.round_steps:.0f} | {point.visited_states} "
            f"| {point.unseen_per_round:.1f} | {point.forced_fraction:.3f} |"
        )
    return "\n".join(lines) + "\n"
