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

from agent_code.tabular_q_agent.config import ENV_VAR, MODEL_ENV_VAR
from agent_code.tabular_q_agent.features import ENCODINGS
from agent_code.tabular_q_agent.metrics import read_records
from agent_code.tabular_q_agent.qtable import QTable
from tournament.runner import run_schedule
from tournament.schedule import CANDIDATE_ARM, PRESETS, build_schedule, seeds
from tournament.stats import ArmSummary, summarise_arm
from tournament.storage import read_rounds, write_rounds
from training.config import LEARNER
from training.driver import METRICS_FILE, environment, load_run, snapshots

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


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def evaluate_run(
    run_dir: Path, *, jobs: int = 1, progress: bool = False
) -> dict[str, list[CurvePoint]]:
    """Evaluate every snapshot on every preset of the run's curriculum."""
    curriculum, _ = load_run(run_dir)
    encoding = ENCODINGS[curriculum.agent_config().encoding]
    metrics_file = run_dir / METRICS_FILE
    records = read_records(metrics_file) if metrics_file.exists() else []

    curves: dict[str, list[CurvePoint]] = {}
    for evaluation in curriculum.evaluations:
        schedule = build_schedule(
            LEARNER,
            PRESETS[evaluation.preset],
            seeds(evaluation.seeds, evaluation.seed_start),
            with_control=False,
        )
        points: list[CurvePoint] = []
        for rounds_trained, snapshot in snapshots(run_dir):
            out = (
                run_dir
                / EVAL_DIR
                / f"{evaluation.preset}_round_{rounds_trained:06d}.jsonl"
            )
            if out.exists():
                results = read_rounds(out)
            else:
                env = {
                    MODEL_ENV_VAR: str(snapshot),
                    ENV_VAR: json.dumps(curriculum.params),
                }
                with environment(env):
                    results = run_schedule(schedule, jobs=jobs, progress=progress)
                write_rounds(out, results)
            window = [
                record
                for record in records
                if rounds_trained - curriculum.eval_every
                < record.rounds_trained
                <= rounds_trained
            ]
            points.append(
                CurvePoint(
                    rounds_trained=rounds_trained,
                    summary=summarise_arm(results, CANDIDATE_ARM),
                    visited_states=QTable.load(snapshot, encoding).visited_states,
                    unseen_per_round=_mean([float(r.unseen_decisions) for r in window]),
                    forced_fraction=_mean([r.forced_fraction for r in window]),
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
    lines = [
        f"## {title}",
        "",
        "| rounds trained | rounds | score [95% CI] | win | survive | suicides "
        "| coins | kills | round len | visited states | unseen / round (train) "
        "| forced (train) |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point in points:
        s = point.summary
        score = f"{s.score.mean:.2f} [{s.score.ci.low:.2f}, {s.score.ci.high:.2f}]"
        lines.append(
            f"| {point.rounds_trained} | {s.rounds} | {score} | {s.win_rate:.2f} "
            f"| {s.survival_rate:.2f} | {s.suicides:.2f} | {s.coins:.2f} "
            f"| {s.kills:.2f} | {s.round_steps:.0f} | {point.visited_states} "
            f"| {point.unseen_per_round:.1f} | {point.forced_fraction:.3f} |"
        )
    return "\n".join(lines) + "\n"
