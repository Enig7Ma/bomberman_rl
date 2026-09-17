"""D6 navigation experiment; no changes to rewards, features or learning rules.

Run as python -m docs.experiments.dqn_d6; all game runs use the existing harness.
The save-hook wrapper only measures probes and archives completed full saves.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import logging
import platform
import random
import shutil
import subprocess
import time
import traceback
from collections.abc import Callable, Sequence
from functools import wraps
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np

from agent_code.dqn_agent.core.world_model import ACTIONS, Observation
from agent_code.dqn_agent.encoder import ENCODERS
from agent_code.dqn_agent.features import ENCODINGS, Extractor
from agent_code.dqn_agent.network import QNetwork
from agent_code.dqn_agent.probe import Probe
from agent_code.dqn_agent.symmetry import canonical, to_canonical
from tournament import engine
from tournament.engine import (
    REPO_ROOT,
    WorldConfig,
    quiet_logging,
    reset_framework_logging,
)
from tournament.results import RoundResult
from tournament.runner import run_schedule
from tournament.schedule import PRESETS, build_schedule
from tournament.storage import write_rounds
from training.config import load_curriculum
from training.driver import environment, load_run, snapshots, train_run
from training.spec import AgentSpec
from training.world import TrainingWorld, create_training_world, play_round

ENCODER = ENCODERS["onehot_e3"]
VALIDATION_SEEDS = tuple(range(500, 510))
COMBINATIONS = ((3e-4, 1000), (3e-4, 250), (1e-4, 1000), (1e-4, 250))


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def version_info() -> dict[str, Any]:
    files = sorted(
        {
            *REPO_ROOT.glob("agent_code/dqn_agent/**/*.py"),
            *REPO_ROOT.glob("training/*.py"),
            Path(__file__).resolve(),
        }
    )
    return {
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "status": subprocess.check_output(["git", "status", "--short"], text=True),
        "source_sha256": {str(p.relative_to(REPO_ROOT)): checksum(p) for p in files},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "torch",
                "pygame",
                "scipy",
                "pytest",
                "ruff",
                "pyright",
            )
        },
    }


def collect_probe(directory: Path) -> Path:
    """1000 rows per source, sampled without replacement from 20 solo games.

    Each source plays seeds 900..909 once. Both Python and legacy NumPy RNG are
    seeded AFTER rule_based setup (which otherwise reseeds NumPy from entropy).
    BFS and feature extraction use separate private RNGs. No learner is created.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "probe.npz"
    if path.exists():
        raise FileExistsError(
            "probe already exists; reuse it or choose a new directory"
        )
    xs: list[Any] = []
    masks: list[Any] = []
    origins: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    old_python, old_numpy = random.getstate(), np.random.get_state()
    try:
        for name in ("bfs_agent", "rule_based_agent"):
            start = len(xs)
            module = importlib.import_module(f"agent_code.{name}.callbacks")
            original = module.act
            for seed in range(900, 910):
                extractor = Extractor(ENCODINGS["E3"], "best_tier", random.Random(seed))

                @wraps(original)
                def capture(
                    agent: object,
                    state: dict[str, Any],
                    extractor: Extractor = extractor,
                    name: str = name,
                    seed: int = seed,
                    original: Callable[[object, dict[str, Any]], str] = original,
                ) -> str:
                    extracted = extractor.extract(Observation.from_game_state(state))
                    index, symmetry = canonical(extracted.features, ENCODINGS["E3"])
                    xs.append(ENCODER.encode(ENCODINGS["E3"].decode(index), extracted))
                    mask = np.zeros(6, dtype=bool)
                    mask[
                        [
                            ACTIONS.index(to_canonical(a, symmetry))
                            for a in extracted.allowed
                        ]
                    ] = True
                    masks.append(mask)
                    origins.append({"agent": name, "seed": seed, "step": state["step"]})
                    return str(original(agent, state))

                with (
                    environment({"BFS_AGENT_PARAMS": json.dumps({"seed": seed})}),
                    quiet_logging(),
                    patch.object(module, "act", capture),
                ):
                    try:
                        world = create_training_world(
                            (name,),
                            "coin-heaven",
                            seed,
                            train_seats=0,
                            log_dir=str(directory / "logs" / f"{name}_{seed}"),
                        )
                        random.seed(seed)
                        np.random.seed(seed)
                        play_round(world)
                    finally:
                        reset_framework_logging()
            counts[name] = len(xs) - start
    finally:
        random.setstate(old_python)
        np.random.set_state(old_numpy)
    rng = np.random.default_rng(900)
    chosen: list[int] = []
    offset = 0
    for count in counts.values():
        if count < 1000:
            raise RuntimeError(
                f"only {count} source states; need 1000 without replacement"
            )
        chosen.extend(
            (np.sort(rng.choice(count, 1000, replace=False)) + offset).tolist()
        )
        offset += count
    x, mask_array = np.stack(xs)[chosen], np.stack(masks)[chosen]
    np.savez_compressed(
        path, x=x, masks=mask_array, schema_id=np.array(ENCODER.schema_id)
    )
    probe = Probe.load(path, ENCODER.schema_id, ENCODER.dim)
    dump(
        directory / "probe_manifest.json",
        {
            "schema_id": ENCODER.schema_id,
            "sha256": checksum(path),
            "fingerprint": probe.fingerprint,
            "scenario": "coin-heaven",
            "collection_seeds": list(range(900, 910)),
            "sample_seed": 900,
            "source_counts": counts,
            "states": len(x),
            "unique_encoded_states": len(np.unique(x, axis=0)),
            "selected_rows": [origins[i] for i in chosen],
            "code": version_info(),
            "sampling": "1000 per agent, without replacement; "
            "repeated encoded states retained",
        },
    )
    return path


def aggregate(results: list[RoundResult]) -> dict[str, Any]:
    successes = [r for r in results if r.focus.coins == 50]
    n = len(results)
    return {
        "rounds": n,
        "mean_coins": sum(r.focus.coins for r in results) / n,
        "all50_fraction": len(successes) / n,
        "mean_steps_all": sum(r.steps for r in results) / n,
        "mean_steps_success": sum(r.steps for r in successes) / len(successes)
        if successes
        else None,
        "max_steps": max(r.steps for r in results),
        "strict_126": len(successes) == n and all(r.steps <= 126 for r in results),
        "deaths": sum(not r.focus.survived for r in results),
        "invalid": sum(r.focus.invalid for r in results),
        "timeouts": sum(r.focus.timeouts for r in results),
    }


def evaluate(directory: Path) -> list[dict[str, Any]]:
    """All snapshots, including zero, on exactly seeds 500..509; no control arm."""
    from agent_code.dqn_agent import callbacks

    artifacts = [
        directory / name for name in ("checkpoint.pt", "replay.npz", "q_net.npz")
    ]
    before = {p.name: checksum(p) for p in artifacts if p.exists()}
    schedule = build_schedule(
        "dqn_agent", PRESETS["coin-heaven-solo"], VALIDATION_SEEDS, with_control=False
    )
    rows: list[dict[str, Any]] = []
    original = callbacks.setup

    def evaluation_world(config: WorldConfig, log_dir: str = "logs") -> TrainingWorld:
        return create_training_world(
            config.lineup, config.scenario, config.seed, train_seats=0, log_dir=log_dir
        )

    for count, model in snapshots(directory):
        expected = QNetwork.load(model, ENCODER)

        @wraps(original)
        def checked_setup(
            agent: object, model: Path = model, expected: QNetwork = expected
        ) -> None:
            handle = cast(Any, agent)
            original(handle)
            assert not handle.train and handle.trainer is None
            assert isinstance(handle.q_function, QNetwork)
            assert handle.model_file.resolve() == model.resolve()
            for key in expected.arrays:
                np.testing.assert_array_equal(
                    handle.q_function.arrays[key], expected.arrays[key]
                )

        with (
            environment(
                {
                    "DQN_AGENT_MODEL": str(model.resolve()),
                    "DQN_AGENT_PARAMS": json.dumps(
                        {"seed": 500, "policy": "learned", "encoder": "onehot_e3"}
                    ),
                }
            ),
            patch.object(callbacks, "setup", checked_setup),
            patch.object(engine, "create_world", evaluation_world),
        ):
            results = run_schedule(
                schedule, jobs=1, log_dir=str(directory / "eval_logs")
            )
        write_rounds(
            directory / "evaluation" / f"transition_{count:09d}.jsonl", results
        )
        archive = directory / "checkpoints" / f"transition_{count:09d}"
        rows.append(
            {
                "transitions": count,
                "snapshot": str(model),
                "snapshot_sha256": checksum(model),
                "checkpoint": str(archive / "checkpoint.pt"),
                "checkpoint_sha256": checksum(archive / "checkpoint.pt"),
                **aggregate(results),
            }
        )
    assert before == {p.name: checksum(p) for p in artifacts if p.exists()}
    dump(directory / "evaluation_summary.json", rows)
    return rows


def run(config_path: Path, directory: Path, seed: int) -> dict[str, Any]:
    """One fresh run or a resume of that exact run; failure remains in results."""
    from agent_code.dqn_agent.train import Trainer
    from training import dqn

    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    course = load_curriculum(config_path)
    summary_path = directory / "run_summary.json"
    if summary_path.exists():
        stored_course, stored_seed = load_run(directory)
        if stored_course != course or stored_seed != seed:
            raise ValueError("existing result belongs to another config or seed")
        cached: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))
        if "evaluation" not in cached:
            cached["evaluation"] = evaluate(directory)
            dump(summary_path, cached)
        return cached  # Do not replace measured training time with a no-op time.
    active: list[Trainer] = []
    original_save, original_world = AgentSpec.save_chunk, dqn.create_training_world

    def world_hook(
        lineup: Sequence[str],
        scenario: str,
        seed: int,
        *,
        train_seats: int = 1,
        log_dir: str = "logs",
    ) -> TrainingWorld:
        world = original_world(
            lineup, scenario, seed, train_seats=train_seats, log_dir=log_dir
        )
        active[:] = [cast(Any, world).agents[0].backend.runner.fake_self.trainer]
        return world

    def save_hook(spec: AgentSpec, agent: object) -> None:
        trainer = cast(Trainer, cast(Any, agent).trainer)
        records = cast(dict[str, Any], trainer.driver_state)["records"]
        at_snapshot = trainer.transitions == 0 or (records and records[-1]["snapshot"])
        diagnostics = None
        if at_snapshot:
            assert trainer.probe is not None
            diagnostics = {
                "transitions": trainer.transitions,
                "updates": trainer.learner.updates,
                "fingerprint": trainer.probe.fingerprint,
                **trainer.probe.measure(trainer.learner.online),
            }
        original_save(spec, agent)
        if at_snapshot:
            count = trainer.transitions
            archive = directory / "checkpoints" / f"transition_{count:09d}"
            archive.mkdir(parents=True, exist_ok=True)
            for name in ("checkpoint.pt", "replay.npz", "q_net.npz"):
                shutil.copyfile(directory / name, archive / name)
            dump(archive / "probe.json", diagnostics)
            target = directory / "snapshots" / f"transition_{count:09d}.npz"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(directory / "q_net.npz", target)

    dump(directory / "environment.json", version_info())
    shutil.copyfile(Path(__file__), directory / "experiment_source.py")
    started = time.perf_counter()
    failure: str | None = None
    try:
        with (
            patch.object(AgentSpec, "save_chunk", save_hook),
            patch.object(dqn, "create_training_world", world_hook),
        ):
            train_run(course, directory, seed)
    except Exception:
        failure = traceback.format_exc()
        (directory / "failure.txt").write_text(failure, encoding="utf-8")
        if active:
            try:
                active[0].learner.export().save(directory / "failed_q_net.npz")
            except (ValueError, FloatingPointError):
                pass
    elapsed = time.perf_counter() - started
    metrics_path = directory / "metrics.jsonl"
    metrics: list[dict[str, Any]] = (
        [json.loads(line) for line in metrics_path.read_text().splitlines()]
        if metrics_path.exists()
        else []
    )
    losses = [
        float(row["mean_loss"]) for row in metrics if row["mean_loss"] is not None
    ]
    early = float(np.mean(losses[:20])) if losses else None
    late = float(np.mean(losses[-20:])) if losses else None
    transitions = (
        active[0].transitions
        if active
        else (metrics[-1]["transitions"] if metrics else 0)
    )
    summary: dict[str, Any] = {
        "directory": str(directory),
        "config": str(config_path),
        "seed": seed,
        "failure": failure,
        "training_seconds": elapsed,
        "transitions": transitions,
        "rounds": len(metrics),
        "updates": metrics[-1]["updates"] if metrics else 0,
        "transitions_per_second": transitions / elapsed,
        "loss_first20": early,
        "loss_last20": late,
        "stable": failure is None
        and transitions >= 50000
        and early is not None
        and late is not None
        and late <= 1.25 * early + 1e-6,
        "probe_sha256": checksum(Path(course.params["probe_path"])),
    }
    dump(directory / "run_summary.json", summary)
    summary["evaluation"] = evaluate(directory)
    dump(directory / "run_summary.json", summary)
    return summary


def select(directories: list[Path]) -> dict[str, Any]:
    """Preregistered final-snapshot ranking; deterministic combination-order tie."""
    rows = [json.loads((p / "run_summary.json").read_text()) for p in directories]
    eligible = [r for r in rows if r["stable"]]
    if not eligible:
        return {"selected": None, "reason": "no stable completed pilot", "pilots": rows}

    def rank(row: dict[str, Any]) -> tuple[float, float, float]:
        final = row["evaluation"][-1]
        return (-final["mean_coins"], -final["all50_fraction"], final["mean_steps_all"])

    return {"selected": min(eligible, key=rank)["directory"], "pilots": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("collect", "run", "evaluate", "select"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pilots", nargs="*", type=Path, default=[])
    args = parser.parse_args()
    if args.command == "collect":
        print(collect_probe(args.out))
    elif args.command == "run":
        if args.config is None:
            parser.error("run requires --config")
        print(json.dumps(run(args.config, args.out, args.seed), indent=2))
    elif args.command == "evaluate":
        print(json.dumps(evaluate(args.out), indent=2))
    else:
        dump(args.out, select(args.pilots))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
