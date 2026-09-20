"""Audit a finished navigation run: build a wider probe and re-read its metrics.

Collects at most 100 rounds of probe states (target 80), then reports probe
measurements, loss windows and target-sync rounds as CSV. It never trains, and
re-running reads the cached collection instead of playing more rounds. Paths
are fixed in ``ROOT`` and ``REPORT``. Run from the repository root::

    uv run python -m docs.experiments.dqn_d6_audit
"""

import csv
import importlib
import json
import random
from collections import Counter
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
from numpy.typing import NDArray

from agent_code.dqn_agent.core.world_model import ACTIONS, Observation
from agent_code.dqn_agent.encoder import OneHotE3
from agent_code.dqn_agent.features import ENCODINGS, Extractor
from agent_code.dqn_agent.network import QNetwork
from agent_code.dqn_agent.probe import Probe
from agent_code.dqn_agent.symmetry import canonical, to_canonical
from docs.experiments.dqn_d6 import checksum, dump, version_info
from tournament.engine import quiet_logging, reset_framework_logging
from training.driver import environment, snapshots
from training.world import create_training_world, play_round

ROOT = Path("results/dqn/d6_20260917")
OUT = Path("results/dqn/d6_audit_20260918")
REPORT = Path("docs/experiments/dqn_d6/audit")
ENCODER = OneHotE3()
CASES = (
    ("coin-heaven", ()),
    ("loot-crate", ()),
    ("classic", ("rule_based_agent",) * 3),
    ("coin-heaven", ("rule_based_agent",) * 3),
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def is_focus(instance_name: str, code_name: str) -> bool:
    """Engine adds _0 to the focus seat when its code appears more than once."""
    return instance_name in (code_name, code_name + "_0")


def collect() -> tuple[NDArray[np.float32], NDArray[np.bool_], list[dict[str, Any]]]:
    cache = OUT / "collection.npz"
    xs: list[NDArray[np.float32]] = []
    masks: list[NDArray[np.bool_]] = []
    origins: list[dict[str, Any]] = []
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            xs = [row.copy() for row in data["x"]]
            masks = [row.copy() for row in data["masks"]]
        origins = json.loads((OUT / "origins.json").read_text())
    elif (OUT / "started.json").exists():
        raise RuntimeError("partial collection; inspect ledger before any more games")
    else:
        dump(
            OUT / "started.json",
            {"planned_rounds": 80, "cap": 100, "code": version_info()},
        )
    covered = {(r["case"], r["agent"], r["seed"]) for r in origins}
    ledger = OUT / "rounds.jsonl"
    played = len(ledger.read_text().splitlines()) if ledger.exists() else 0
    missing = 80 - len(covered)
    if played + missing > 100:
        raise RuntimeError("collection would exceed 100 additional rounds")
    initial_bytes = [x.tobytes() + m.tobytes() for x, m in zip(xs, masks, strict=True)]
    old_py, old_np = random.getstate(), np.random.get_state()
    try:
        for case, (scenario, opponents) in enumerate(CASES):
            for name in ("bfs_agent", "rule_based_agent"):
                module = importlib.import_module(f"agent_code.{name}.callbacks")
                original: Callable[..., str] = module.act
                for seed in range(900, 910):
                    if (case, name, seed) in covered:
                        continue
                    extractor = Extractor(
                        ENCODINGS["E3"], "best_tier", random.Random(seed)
                    )

                    @wraps(original)
                    def capture(
                        agent: object,
                        state: dict[str, Any],
                        extractor: Extractor = extractor,
                        name: str = name,
                        seed: int = seed,
                        case: int = case,
                        scenario: str = scenario,
                        opponents: tuple[str, ...] = opponents,
                        original: Callable[..., str] = original,
                    ) -> str:
                        # Only the focus seat; a module can also serve opponents.
                        if not is_focus(state["self"][0], name):
                            return original(agent, state)
                        extracted = extractor.extract(
                            Observation.from_game_state(state)
                        )
                        index, symmetry = canonical(extracted.features, ENCODINGS["E3"])
                        x = ENCODER.encode(ENCODINGS["E3"].decode(index), extracted)
                        mask = np.zeros(6, dtype=bool)
                        mask[
                            [
                                ACTIONS.index(to_canonical(a, symmetry))
                                for a in extracted.allowed
                            ]
                        ] = True
                        xs.append(x)
                        masks.append(mask)
                        initial_bytes.append(x.tobytes() + mask.tobytes())
                        action = original(agent, state)
                        origins.append(
                            {
                                "case": case,
                                "scenario": scenario,
                                "opponents": len(opponents),
                                "agent": name,
                                "seed": seed,
                                "step": state["step"],
                                "action": action,
                                "canonical_action": to_canonical(action, symmetry),
                                "raw_features": extracted.features.values(),
                            }
                        )
                        return action

                    with (
                        environment({"BFS_AGENT_PARAMS": json.dumps({"seed": seed})}),
                        quiet_logging(),
                        patch.object(module, "act", capture),
                    ):
                        try:
                            world = create_training_world(
                                (name, *opponents),
                                scenario,
                                seed,
                                train_seats=0,
                                log_dir=str(OUT / "logs" / f"{case}_{name}_{seed}"),
                            )
                            random.seed(seed)
                            np.random.seed(seed)
                            play_round(world)
                            with (OUT / "rounds.jsonl").open(
                                "a", encoding="utf-8"
                            ) as file:
                                file.write(
                                    json.dumps(
                                        {"case": case, "agent": name, "seed": seed}
                                    )
                                    + "\n"
                                )
                        finally:
                            reset_framework_logging()
    finally:
        random.setstate(old_py)
        np.random.set_state(old_np)
    assert len({id(x) for x in xs}) == len(xs)
    assert len({id(x) for x in masks}) == len(masks)
    assert all(
        x.flags.owndata and m.flags.owndata for x, m in zip(xs, masks, strict=True)
    )
    assert initial_bytes == [
        x.tobytes() + m.tobytes() for x, m in zip(xs, masks, strict=True)
    ]
    x, mask_array = np.stack(xs), np.stack(masks)
    np.savez_compressed(cache, x=x, masks=mask_array)
    dump(OUT / "origins.json", origins)
    return x, mask_array, origins


def describe(
    x: NDArray[np.float32],
    masks: NDArray[np.bool_],
    origins: list[dict[str, Any]],
    label: str,
) -> None:
    unique, counts = np.unique(x, axis=0, return_counts=True)
    write_csv(
        REPORT / f"{label}_states.csv",
        [
            {"count": int(count), **ENCODER.decode(row).values()}
            for row, count in zip(unique, counts, strict=True)
        ],
    )
    groups: dict[str, list[int]] = {}
    for i, row in enumerate(origins):
        key = f"{row['case']}:{row['agent']}:{row['seed']}"
        groups.setdefault(key, []).append(i)
    write_csv(
        REPORT / f"{label}_coverage.csv",
        [
            {
                "group": key,
                "scenario": origins[ids[0]]["scenario"],
                "opponents": origins[ids[0]]["opponents"],
                "count": len(ids),
                "min_step": min(origins[i]["step"] for i in ids),
                "max_step": max(origins[i]["step"] for i in ids),
                "unique_states": len(np.unique(x[ids], axis=0)),
            }
            for key, ids in groups.items()
        ],
    )
    features = [ENCODER.decode(row).values() for row in x]
    dump(
        REPORT / f"{label}_composition.json",
        {
            "observations": len(x),
            "unique_canonical": len(unique),
            "unique_raw_features": len(
                {json.dumps(r["raw_features"], sort_keys=True) for r in origins}
            ),
            "actions": dict(Counter(r["action"] for r in origins)),
            "canonical_actions": dict(Counter(r["canonical_action"] for r in origins)),
            "allowed_action_counts": dict(
                zip(ACTIONS, masks.sum(axis=0).tolist(), strict=True)
            ),
            "features": {
                field: dict(Counter(str(r[field]) for r in features))
                for field in features[0]
            },
        },
    )


def probes() -> None:
    x, masks, origins = collect()
    old_path = ROOT / "probe/probe.npz"
    old_hash = checksum(old_path)
    manifest = json.loads((ROOT / "probe/probe_manifest.json").read_text())
    lookup = {
        (r["agent"], r["seed"], r["step"]): i
        for i, r in enumerate(origins)
        if r["case"] == 0
    }
    selected = [
        lookup[(r["agent"], r["seed"], r["step"])] for r in manifest["selected_rows"]
    ]
    with np.load(old_path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["x"], x[selected])
        np.testing.assert_array_equal(data["masks"], masks[selected])
    describe(x[selected], masks[selected], [origins[i] for i in selected], "original")
    # Equal representation of four game cases and two focus agents. Random
    # samples cover entire trajectories; no uniqueness quota or replacement.
    rng = np.random.default_rng(909)
    chosen: list[int] = []
    for case in range(4):
        for name in ("bfs_agent", "rule_based_agent"):
            ids = [
                i
                for i, r in enumerate(origins)
                if r["case"] == case and r["agent"] == name
            ]
            if len(ids) < 250:
                raise RuntimeError(
                    f"insufficient coverage {case}/{name}: {len(ids)}; no extra games"
                )
            chosen.extend(sorted(rng.choice(ids, size=250, replace=False).tolist()))
    probe = Probe(x[chosen], masks[chosen])
    path = OUT / "probe_v2.npz"
    np.savez_compressed(
        path, x=probe.x, masks=probe.masks, schema_id=np.array(ENCODER.schema_id)
    )
    dump(
        OUT / "probe_v2_manifest.json",
        {
            "schema_id": ENCODER.schema_id,
            "sha256": checksum(path),
            "fingerprint": probe.fingerprint,
            "original_sha256": old_hash,
            "rounds": len((OUT / "rounds.jsonl").read_text().splitlines()),
            "sampling_seed": 909,
            "selected_rows": [origins[i] for i in chosen],
            "sampling": "250 per case and focus agent, without replacement; "
            "repeated encodings retained",
        },
    )
    describe(probe.x, probe.masks, [origins[i] for i in chosen], "v2")
    rows: list[dict[str, Any]] = []
    for summary in sorted(ROOT.glob("**/run_summary.json")):
        probe.previous = None
        for count, model in snapshots(summary.parent):
            net = QNetwork.load(model, ENCODER)
            q = np.stack([net.values(row) for row in probe.x])
            allowed = q[probe.masks]
            violation = not np.isfinite(q).all() or bool((np.abs(q) > 50).any())
            # Do not hide offending snapshots: keep values and a failed flag.
            actions = np.where(probe.masks, q, -np.inf).argmax(axis=1)
            churn = (
                None
                if probe.previous is None
                else float(
                    np.count_nonzero(actions != np.array(probe.previous)) / len(actions)
                )
            )
            probe.previous = actions.tolist()
            rows.append(
                {
                    "run": str(summary.parent.relative_to(ROOT)),
                    "transitions": count,
                    "snapshot": str(model),
                    "sha256": checksum(model),
                    "violation": violation,
                    "allowed_mean_q": float(allowed.mean()),
                    "allowed_min_q": float(allowed.min()),
                    "allowed_max_q": float(allowed.max()),
                    "max_abs_q": float(np.abs(q).max()),
                    "policy_churn": churn,
                }
            )
    write_csv(REPORT / "v2_snapshots.csv", rows)
    assert checksum(old_path) == old_hash


def loss_window(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = sum(r["updates_this_round"] for r in rows)
    result: dict[str, Any] = {
        "round_first": rows[0]["rounds_trained"],
        "round_last": rows[-1]["rounds_trained"],
        "measurements": len(rows),
        "updates": count,
        "update_first": rows[0]["updates"] - rows[0]["updates_this_round"] + 1,
        "update_last": rows[-1]["updates"],
    }
    for field in ("mean_loss", "mean_abs_td", "mean_grad_norm"):
        result[field] = float(np.mean([r[field] for r in rows]))
        result[field + "_weighted"] = (
            sum(r[field] * r["updates_this_round"] for r in rows) / count
        )
    return result


def saved_metrics() -> None:
    summaries: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    evaluation: list[dict[str, Any]] = []
    for path in sorted(ROOT.glob("**/run_summary.json")):
        run = str(path.parent.relative_to(ROOT))
        summary = json.loads(path.read_text())
        config = json.loads((path.parent / "run.json").read_text())["curriculum"][
            "params"
        ]
        target = config["target_every"]
        all_rows = [
            json.loads(line)
            for line in (path.parent / "metrics.jsonl").read_text().splitlines()
        ]
        rows = [r for r in all_rows if r["mean_loss"] is not None]
        assert all(r["updates_this_round"] > 0 for r in rows)
        assert all(
            r["updates_this_round"] == 0 for r in all_rows if r["mean_loss"] is None
        )
        early, late = loss_window(rows[:20]), loss_window(rows[-20:])
        for label, window in (("early", early), ("late", late)):
            windows.append({"run": run, "window": label, **window})
        threshold = 1.25 * early["mean_loss"] + 1e-6
        assert np.isclose(early["mean_loss"], summary["loss_first20"])
        assert np.isclose(late["mean_loss"], summary["loss_last20"])
        deltas: dict[bool, list[float]] = {True: [], False: []}
        for prev, row in zip(rows, rows[1:], strict=False):
            start = row["updates"] - row["updates_this_round"] + 1
            # Updates use target copied AFTER each multiple. A sync at start-1
            # affects this whole round; a sync at end affects only the next one.
            affected = (start - 1 + target - 1) // target <= (
                row["updates"] - 1
            ) // target
            delta = row["mean_loss"] - prev["mean_loss"]
            deltas[affected].append(delta)
            changes.append(
                {
                    "run": run,
                    "round": row["rounds_trained"],
                    "update_first": start,
                    "update_last": row["updates"],
                    "fresh_sync_target_in_round": affected,
                    "loss": row["mean_loss"],
                    "delta_loss": delta,
                    "td": row["mean_abs_td"],
                    "grad_norm": row["mean_grad_norm"],
                }
            )
        summaries.append(
            {
                "run": run,
                "updated_rounds": len(rows),
                "no_update_rounds": len(all_rows) - len(rows),
                "early": early["mean_loss"],
                "late": late["mean_loss"],
                "threshold": threshold,
                "ratio": late["mean_loss"] / early["mean_loss"],
                "passed": late["mean_loss"] <= threshold,
                "sync_rounds": len(deltas[True]),
                "nonsync_rounds": len(deltas[False]),
                "mean_delta_sync": float(np.mean(deltas[True])),
                "mean_delta_other": float(np.mean(deltas[False])),
            }
        )
        final = summary["evaluation"][-1]
        records = [
            json.loads(line)
            for line in (
                path.parent
                / "evaluation"
                / f"transition_{final['transitions']:09d}.jsonl"
            )
            .read_text()
            .splitlines()
        ]
        steps = [r["steps"] for r in records]
        evaluation.append(
            {
                "run": run,
                "seed": summary["seed"],
                "mean": float(np.mean(steps)),
                "min": min(steps),
                "max": max(steps),
                "fraction_le126": sum(n <= 126 for n in steps) / len(steps),
                "steps_in_seed_order": json.dumps(steps),
            }
        )
    write_csv(REPORT / "loss_filter.csv", summaries)
    write_csv(REPORT / "loss_windows.csv", windows)
    write_csv(REPORT / "target_sync_rounds.csv", changes)
    write_csv(REPORT / "final_steps.csv", evaluation)


def main() -> None:
    saved_metrics()
    probes()


if __name__ == "__main__":
    main()
