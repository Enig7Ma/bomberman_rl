"""Matched navigation/crates study, three seeds; no hunting or held-out games."""

import argparse
import importlib
import json
import multiprocessing as mp
import random
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from functools import wraps
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
from numpy.typing import NDArray

from agent_code.dqn_agent.core.world_model import ACTIONS
from agent_code.dqn_agent.encoder import OneHotE3
from agent_code.dqn_agent.network import QNetwork
from agent_code.dqn_agent.probe import Probe
from agent_code.tabular_q_agent.features import ENCODINGS, Features
from agent_code.tabular_q_agent.qtable import QTable
from docs.experiments.dqn_candidates import rule_streams
from docs.experiments.dqn_d6 import checksum, dump, version_info
from tournament import engine
from tournament.engine import WorldConfig, quiet_logging
from tournament.schedule import PRESETS
from training.config import load_curriculum
from training.driver import environment, read_chunk_records, train_run
from training.spec import AgentSpec
from training.world import TrainingWorld, create_training_world

PROBE = Path("results/dqn/d6_audit_20260918/probe_v2.npz")
PROBE_SHA = "008253a6aa52af1e006b0d3858591f4a0cc94c03729073e659fb7ce010e3189f"
ENCODER = OneHotE3()
AGENTS = ("tabular_q_agent", "dqn_agent")
PRESET_NAMES = ("coin-heaven-solo", "crates-solo", "vs-rule-based")


class TableFunction:
    def __init__(self, table: QTable) -> None:
        self.table = table

    def values(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        state = Features(**asdict(ENCODER.decode(x)))
        return self.table.q[ENCODINGS["E3"].encode(state)].astype(np.float32)


def train_worker(
    root: str, name: str, seed: int, smoke: bool, *, resume: bool = False
) -> dict[str, Any]:
    directory = Path(root) / name / f"run_{seed}"
    course = load_curriculum(Path(f"docs/experiments/d8_reduced_{name}.json"))
    if smoke:
        course = replace(
            course,
            chunk_rounds=1,
            stages=tuple(replace(s, transitions=401, rounds=0) for s in course.stages),
            params={**course.params, **({"warmup": 8} if name == "dqn_agent" else {})},
        )
    probe = Probe.load(PROBE, ENCODER.schema_id, ENCODER.dim)
    points: list[dict[str, Any]] = []
    prior_seconds = 0.0
    recovered_from: str | None = None
    if resume:
        previous = json.loads((directory / "training_summary.json").read_text())
        if name != "tabular_q_agent" or "PermissionError" not in (
            previous["failure"] or ""
        ):
            raise ValueError(
                "only explicit recovery of tabular I/O failure is supported"
            )
        incident = directory / "recovery"
        incident.mkdir(exist_ok=False)
        shutil.copy2(directory / "q_table.npz", incident / "committed_q_table.npz")
        shutil.copy2(
            directory / "training_summary.json", incident / "interrupted_summary.json"
        )
        shutil.copy2(directory / "failure.txt", incident / "failure.txt")
        shutil.copy2(Path(__file__), incident / "harness.py.txt")
        shutil.copy2(
            Path("training/tabular_steps.py"), incident / "tabular_steps.py.txt"
        )
        points = json.loads((directory / "points.json").read_text())
        prior_seconds = float(previous["seconds"])
        recovered_from = str(incident)
    started = time.perf_counter()

    def archive(table: QTable | None = None) -> None:
        records = read_chunk_records(directory)
        # DQN's save hook precedes publish; its cursor is authoritative.
        if table is None:
            from agent_code.dqn_agent.persistence import load_checkpoint

            cp = load_checkpoint(directory / "checkpoint.pt")
            raw = cp["driver_state"]["records"]
            count, updates = cp["transitions"], cp["learner"]["updates"]
            take = not raw or bool(raw[-1]["snapshot"])
            network = QNetwork.load(directory / "q_net.npz", ENCODER)
        else:
            count = int(table.meta.get("steps_trained", 0))
            updates = count
            take = not records or bool(records[-1].snapshot)
            network = TableFunction(table)
        if not take or any(r["transitions"] == count for r in points):
            return
        target = directory / "archive" / f"transition_{count:09d}"
        target.mkdir(parents=True)
        files = (
            ("q_net.npz", "checkpoint.pt", "replay.npz")
            if table is None
            else ("q_table.npz",)
        )
        for f in files:
            shutil.copy2(directory / f, target / f)
        # Preserve the failing artifact too if this raises; never retry a Q stop.
        measured = probe.measure(network)
        points.append(
            {
                "transitions": count,
                "updates": updates,
                "seconds": prior_seconds + time.perf_counter() - started,
                "directory": str(target),
                "probe": measured,
                "hashes": {f: checksum(target / f) for f in files},
            }
        )
        dump(directory / "points.json", points)

    failure = None
    try:
        if name == "dqn_agent":
            original = AgentSpec.save_chunk

            def save(spec: AgentSpec, agent: object) -> None:
                original(spec, agent)
                archive()

            with patch.object(AgentSpec, "save_chunk", save):
                train_run(course, directory, seed)
        else:
            from training import tabular_steps

            original_publish = tabular_steps.publish

            def publish(path: Path, table: QTable) -> None:
                original_publish(path, table)
                archive(table)

            with patch.object(tabular_steps, "publish", publish):
                train_run(course, directory, seed)
    except Exception:
        failure = traceback.format_exc()
        (directory / "failure.txt").write_text(failure, encoding="utf-8")
    elapsed = prior_seconds + time.perf_counter() - started
    chunks = read_chunk_records(directory)
    summary = {
        "agent": name,
        "seed": seed,
        "smoke": smoke,
        "failure": failure,
        "recovered_from": recovered_from,
        "seconds": elapsed,
        "points": points,
        "transitions": chunks[-1].total_transitions if chunks else 0,
        "updates": chunks[-1].total_updates if chunks else 0,
    }
    dump(directory / "training_summary.json", summary)
    print(
        name,
        seed,
        summary["transitions"],
        "stopped" if failure else "trained",
        flush=True,
    )
    return summary


def evaluate_model(
    model: Path,
    name: str,
    peer_table: Path,
    out: Path,
    *,
    smoke: bool,
    policy: str = "learned",
) -> dict[str, Any]:
    model, peer_table, out = model.resolve(), peer_table.resolve(), out.resolve()
    identity = {
        "model_sha": checksum(model),
        "table_sha": checksum(peer_table),
        "agent": name,
        "policy": policy,
        "smoke": smoke,
    }
    if (out / "summary.json").exists():
        saved = json.loads((out / "summary.json").read_text())
        assert saved["identity"] == identity
        return saved
    peer = QTable.load(peer_table, ENCODINGS["E3"])
    expected = (
        QNetwork.load(model, ENCODER)
        if name == "dqn_agent"
        else QTable.load(model, ENCODINGS["E3"])
    )
    module = importlib.import_module(f"agent_code.{name}.callbacks")
    features = importlib.import_module(f"agent_code.{name}.features")
    sym = importlib.import_module(f"agent_code.{name}.symmetry")
    original_extract = features.Extractor.extract
    prefix = "DQN_AGENT" if name == "dqn_agent" else "TABULAR_Q_AGENT"
    extracted: list[Any] = []

    def extract(self: object, observation: object) -> object:
        result = original_extract(self, observation)
        extracted[:] = [result]
        return result

    original_act = module.act
    rows: list[dict[str, Any]] = []
    diagnostics = {
        b: {"decisions": 0, "greedy_agreement": 0, "base_score": 0.0}
        for b in ("0", "1-9", "10+")
    }
    for preset in PRESET_NAMES:
        for seed in range(
            500, 502 if smoke else (525 if preset == "vs-rule-based" else 510)
        ):
            trace: list[dict[str, Any]] = []
            board: list[str] = []

            @wraps(original_act)
            def act(
                agent: object,
                state: dict[str, Any],
                trace: list[dict[str, Any]] = trace,
                board: list[str] = board,
            ) -> str:
                result = str(original_act(agent, state))
                h = cast(Any, agent)
                ext = extracted[0]
                index, transform = sym.canonical(ext.features, h.encoding)
                visits = int(peer.n[index].sum())
                bucket = "0" if visits == 0 else "1-9" if visits < 10 else "10+"
                allowed = [
                    ACTIONS.index(sym.to_canonical(a, transform)) for a in ext.allowed
                ]
                chosen = ACTIONS.index(sym.to_canonical(result, transform))
                agrees = peer.q[index, chosen] == max(peer.q[index, a] for a in allowed)
                trace.append(
                    {
                        "bucket": bucket,
                        "agreement": bool(agrees),
                        "score": state["self"][1],
                        "action": result,
                    }
                )
                if not board:
                    import hashlib

                    board.append(
                        hashlib.sha256(
                            np.asarray(state["field"]).tobytes()
                            + repr(
                                (
                                    state["self"][3],
                                    state["coins"],
                                    [o[3] for o in state["others"]],
                                )
                            ).encode()
                        ).hexdigest()
                    )
                return result

            def world(config: WorldConfig, log_dir: str = "logs") -> TrainingWorld:
                w = create_training_world(
                    config.lineup,
                    config.scenario,
                    config.seed,
                    train_seats=0,
                    log_dir=log_dir,
                )
                h = cast(Any, w).agents[0].backend.runner.fake_self
                assert not h.train and h.trainer is None
                assert h.model_file.resolve() == model and h.config.policy == policy
                assert h.config.mask == "best_tier"
                assert h.rng.getstate() == random.Random(500).getstate()
                if isinstance(expected, QNetwork):
                    assert isinstance(h.q_function, QNetwork)
                    for key in expected.arrays:
                        np.testing.assert_array_equal(
                            expected.arrays[key], h.q_function.arrays[key]
                        )
                else:
                    np.testing.assert_array_equal(expected.q, h.table.q)
                    np.testing.assert_array_equal(expected.n, h.table.n)
                return w

            with (
                rule_streams(seed, False),
                quiet_logging(),
                environment(
                    {
                        f"{prefix}_MODEL": str(model),
                        f"{prefix}_PARAMS": json.dumps({"seed": 500, "policy": policy}),
                    }
                ),
                patch.object(features.Extractor, "extract", extract),
                patch.object(module, "act", act),
                patch.object(engine, "create_world", world),
            ):
                target = PRESETS[preset]
                result = engine.run_round(
                    WorldConfig((name, *target.opponents), target.scenario, seed),
                    log_dir=str(out / "logs" / f"{preset}_{seed}"),
                )
            for i, item in enumerate(trace):
                following = (
                    trace[i + 1]["score"] if i + 1 < len(trace) else result.focus.score
                )
                bucket = diagnostics[item["bucket"]]
                bucket["decisions"] += 1
                bucket["greedy_agreement"] += int(item["agreement"])
                bucket["base_score"] += following - item["score"]
            rows.append(
                {
                    "preset": preset,
                    "seed": seed,
                    "board": board[0],
                    "wait": sum(r["action"] == "WAIT" for r in trace),
                    **asdict(result.focus),
                }
            )
    assert (
        checksum(model) == identity["model_sha"]
        and checksum(peer_table) == identity["table_sha"]
    )
    result_summary = {"identity": identity, "rows": rows, "visit_bins": diagnostics}
    dump(out / "summary.json", result_summary)
    return result_summary


def evaluate_worker(root: str, summary: dict[str, Any], smoke: bool) -> dict[str, Any]:
    name, seed = summary["agent"], summary["seed"]
    tables = json.loads(
        (Path(root) / "tabular_q_agent" / f"run_{seed}" / "points.json").read_text()
    )
    evaluated: list[dict[str, Any]] = []
    for i, point in enumerate(summary["points"]):
        # Match milestone ordinal; retain actual counts for both learners.
        if i >= len(tables):
            break
        folder = Path(point["directory"])
        peer = Path(tables[i]["directory"]) / "q_table.npz"
        model = folder / ("q_net.npz" if name == "dqn_agent" else "q_table.npz")
        result = evaluate_model(model, name, peer, folder / "evaluation", smoke=smoke)
        evaluated.append(
            {**point, "table_transitions": tables[i]["transitions"], **result}
        )
    output = {**summary, "evaluations": evaluated}
    dump(Path(root) / name / f"run_{seed}" / "evaluated.json", output)
    print(name, seed, "evaluated", flush=True)
    return output


def run(root: Path, smoke: bool, jobs: int) -> None:
    root = root.resolve()
    if root.exists():
        raise FileExistsError("fresh study directory required")
    assert checksum(PROBE) == PROBE_SHA
    root.mkdir(parents=True)
    dump(root / "environment.json", version_info())
    shutil.copy2(Path(__file__), root / "experiment_source.py.txt")
    for name in AGENTS:
        shutil.copy2(
            Path(f"docs/experiments/d8_reduced_{name}.json"), root / f"{name}.json"
        )
    seeds = (0,) if smoke else (0, 1, 2)
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=jobs, mp_context=mp.get_context("spawn")
    ) as pool:
        tasks = [
            pool.submit(train_worker, str(root), a, s, smoke)
            for s in seeds
            for a in AGENTS
        ]
        trained = [task.result() for task in tasks]
        dump(root / "trained.json", trained)
        tasks = [pool.submit(evaluate_worker, str(root), s, smoke) for s in trained]
        evaluated = [task.result() for task in tasks]
    initial = Path(root / "dqn_agent/run_0/points.json")
    points = json.loads(initial.read_text())
    tables = json.loads((root / "tabular_q_agent/run_0/points.json").read_text())
    control = evaluate_model(
        Path(points[0]["directory"]) / "q_net.npz",
        "dqn_agent",
        Path(tables[0]["directory"]) / "q_table.npz",
        root / "safe_random",
        smoke=smoke,
        policy="random",
    )
    dump(
        root / "study.json",
        {
            "smoke": smoke,
            "jobs": jobs,
            "seconds": time.perf_counter() - started,
            "runs": evaluated,
            "safe_random": control,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--jobs", type=int, default=2, choices=(1, 2))
    args = parser.parse_args()
    run(args.out, args.smoke, args.jobs)
