"""Command line for running and reporting tournaments.

``run`` plays a schedule and appends raw per-round records; ``report`` reads
those records back. Keeping them apart means re-analysis never replays games,
which matters when a full comparison is a quarter of an hour of compute.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import settings
from tournament import report
from tournament.runner import run_schedule
from tournament.schedule import PRESETS, build_schedule, seeds
from tournament.storage import read_rounds, write_rounds


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--candidate", required=True, help="agent_code directory to measure"
    )
    parser.add_argument(
        "--preset",
        required=True,
        choices=sorted(PRESETS),
        help="named comparison to run",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=100,
        help="seeds to play; each is played once per seat, per arm",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="first seed - use a disjoint range for held-out evaluation",
    )
    parser.add_argument("--jobs", type=int, default=1, help="worker processes")
    parser.add_argument("--out", type=Path, required=True, help="JSONL to write")
    parser.add_argument(
        "--append", action="store_true", help="add to --out instead of replacing it"
    )
    parser.add_argument(
        "--no-control",
        action="store_true",
        help="skip the paired control arm (halves the work, loses the pairing)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="no progress bar, no report"
    )


def _add_report_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("path", type=Path, help="JSONL written by `run`")
    parser.add_argument("--title", default="", help="heading for the report")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tournament", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_run_arguments(subparsers.add_parser("run", help="play a schedule"))
    _add_report_arguments(subparsers.add_parser("report", help="summarise results"))
    return parser


def _command_run(args: argparse.Namespace) -> int:
    preset = PRESETS[args.preset]
    schedule = build_schedule(
        args.candidate,
        preset,
        seeds(args.rounds, start=args.seed_start),
        with_control=not args.no_control,
    )
    if not args.quiet:
        print(
            f"{args.preset}: {len(schedule)} rounds"
            f" ({args.rounds} seeds x {preset.seats} seats"
            f"{'' if args.no_control else ' x 2 arms'})"
        )

    results = run_schedule(schedule, jobs=args.jobs, progress=not args.quiet)
    write_rounds(args.out, results, append=args.append)

    if not args.quiet:
        print(f"\nwrote {len(results)} rounds to {args.out}\n")
        print(report.render(results, budget=settings.TIMEOUT, title=args.preset))
    return 0


def _command_report(args: argparse.Namespace) -> int:
    results = read_rounds(args.path)
    print(report.render(results, budget=settings.TIMEOUT, title=args.title))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return _command_run(args)
    return _command_report(args)
