"""Command line: ``python -m training run`` and ``python -m training evaluate``.

::

    # three independent runs of one curriculum, in parallel
    uv run python -m training run --curriculum docs/experiments/dqn_d5/smoke.json \\
        --out results/dqn/smoke --runs 3 --jobs 3

    # learning curves for every run under a directory (or for one run)
    uv run python -m training evaluate results/dqn/smoke --jobs 4

    # average each tabular run's 3 newest snapshots into
    # averaged/last3_round_<N>.npz
    uv run python -m training average results/tabular_q/coins --last 3

Run from the repository root. Running ``run`` again on the same ``--out``
resumes unfinished runs.
"""

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

from training.average import average_run
from training.config import load_curriculum
from training.driver import RUN_FILE, read_chunk_records, run_many
from training.evaluate import evaluate_run, render_curve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m training",
        description="Train and evaluate tabular or DQN agents through curricula.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="train (or resume) independent runs")
    run.add_argument("--curriculum", type=Path, required=True)
    run.add_argument(
        "--out", type=Path, required=True, help="parent directory of the runs"
    )
    run.add_argument("--runs", type=int, default=1, help="number of independent runs")
    run.add_argument("--seed", type=int, default=0, help="run seed of the first run")
    run.add_argument(
        "--jobs", type=int, default=0, help="parallel processes (default: one per run)"
    )
    run.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="start every new run from a copy of this inference model",
    )
    run.add_argument("--no-progress", action="store_true")

    evaluate = commands.add_parser("evaluate", help="evaluate every snapshot")
    evaluate.add_argument(
        "path", type=Path, help="a run directory, or a parent of runs"
    )
    evaluate.add_argument("--jobs", type=int, default=1)
    evaluate.add_argument("--no-progress", action="store_true")

    average = commands.add_parser(
        "average", help="average each run's newest snapshots into one table"
    )
    average.add_argument("path", type=Path, help="a run directory, or a parent of runs")
    average.add_argument("--last", type=int, default=3, help="snapshots to average")
    return parser


def render_run_summary(run_dirs: Sequence[Path]) -> str:
    lines = [
        "| run | chunks | rounds trained | wall | rounds/s | steps/s "
        "| visited states |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run_dir in run_dirs:
        records = read_chunk_records(run_dir)
        if not records:
            lines.append(f"| {run_dir.name} | 0 | 0 | - | - | - | - |")
            continue
        wall = sum(record.wall_time for record in records)
        rounds = sum(record.rounds for record in records)
        steps = sum(record.engine_steps for record in records)
        rate = rounds / wall if wall else math.nan
        step_rate = steps / wall if wall else math.nan
        last = records[-1]
        lines.append(
            f"| {run_dir.name} | {len(records)} | {last.rounds_trained} | {wall:.0f} s "
            f"| {rate:.2f} | {step_rate:.0f} | {last.visited_states} |"
        )
    return "\n".join(lines)


def _command_run(args: argparse.Namespace) -> int:
    curriculum = load_curriculum(args.curriculum)
    run_seeds = list(range(args.seed, args.seed + args.runs))
    jobs = args.jobs or len(run_seeds)
    run_dirs = run_many(
        curriculum,
        args.out,
        run_seeds,
        jobs=jobs,
        init_from=args.init_from,
        progress=not args.no_progress,
    )
    print(render_run_summary(list(run_dirs.values())))
    return 0


def _find_runs(path: Path) -> list[Path]:
    """``path`` itself if it is a run directory, else the runs directly below it."""
    if (path / RUN_FILE).exists():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir() if (p / RUN_FILE).exists())
    return []


def _command_average(args: argparse.Namespace) -> int:
    run_dirs = _find_runs(args.path)
    if not run_dirs:
        print(f"no training runs under {args.path}")
        return 2
    for run_dir in run_dirs:
        print(average_run(run_dir, args.last))
    return 0


def _command_evaluate(args: argparse.Namespace) -> int:
    path: Path = args.path
    run_dirs = _find_runs(path)
    if not run_dirs:
        print(f"no training runs under {path}")
        return 2
    for run_dir in run_dirs:
        curves = evaluate_run(run_dir, jobs=args.jobs, progress=not args.no_progress)
        for preset, points in curves.items():
            print(render_curve(points, f"{run_dir.name}: {preset}"))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return _command_run(args)
    if args.command == "average":
        return _command_average(args)
    return _command_evaluate(args)
