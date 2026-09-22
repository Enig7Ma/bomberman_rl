"""Offline D12 integrity checks, clustered intervals and preregistered selection."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from docs.experiments.d12_heldout import PROTOCOL, schedule, sha, write

METRICS = (
    "score",
    "coins",
    "kills",
    "suicides",
    "survived",
    "crates",
    "bombs",
    "invalid",
    "timeouts",
    "steps",
    "moves",
)


def interval(values: NDArray[np.float64]) -> list[float]:
    """One value per world seed: all seats remain in their map cluster."""
    rng = np.random.default_rng(20260921)
    samples = values[rng.integers(len(values), size=(20000, len(values)))].mean(axis=1)
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def choose(dqn: dict[str, Any], table: dict[str, Any]) -> tuple[str, str]:
    dci, tci = dqn["delta_ci"], table["delta_ci"]
    if max(dci[0], tci[0]) <= min(dci[1], tci[1]):
        if dqn["suicides_total"] < table["suicides_total"]:
            return "dqn_stage2_baseline", "overlapping delta CIs; fewer suicides"
        if dqn["suicides_total"] > table["suicides_total"]:
            return "tabular_blended", "overlapping delta CIs; fewer suicides"
        return (
            "tabular_blended",
            "overlapping delta CIs and tied suicides; simpler table",
        )
    winner = (
        "dqn_stage2_baseline" if dqn["delta"] > table["delta"] else "tabular_blended"
    )
    return winner, "non-overlapping delta CIs; larger paired delta"


def summarise(out: Path) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text())
    completed = json.loads((out / "heldout/complete.json").read_text())
    assert completed["protocol_sha256"] == sha(PROTOCOL)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    boards: dict[tuple[str, int, int], str] = {}
    head: list[dict[str, Any]] = []
    evidence = hashlib.sha256()
    for task in schedule(protocol, "heldout"):
        path = out / "heldout/games" / f"{task['id']}.json"
        data = path.read_bytes()
        evidence.update(task["id"].encode() + hashlib.sha256(data).digest())
        row = json.loads(data)
        assert row["status"] == "complete"
        for key in ("seed", "seat", "preset", "candidate", "models", "identities"):
            assert row[key] == task[key]
        assert row["lineup"] == list(task["lineup"])
        key = (row["preset"], row["seed"], row["seat"])
        assert (
            boards.setdefault(key, row["initial_board_sha256"])
            == row["initial_board_sha256"]
        )
        if row["candidate"] == "head_to_head":
            head.append({"seed": row["seed"], "agents": row["result"]["agents"]})
        else:
            focus = row["result"]["agents"][row["seat"]]
            grouped[(row["candidate"], row["preset"])].append(
                {"seed": row["seed"], "seat": row["seat"], **focus}
            )
    summaries: dict[str, dict[str, Any]] = defaultdict(dict)
    maps: dict[tuple[str, str], NDArray[np.float64]] = {}
    for (name, preset), rows in grouped.items():
        per_map = np.array(
            [
                np.mean([r["score"] for r in rows if r["seed"] == s])
                for s in protocol["seeds"]
            ],
            dtype=np.float64,
        )
        maps[(name, preset)] = per_map
        summaries[name][preset] = {
            "games": len(rows),
            "maps": len(per_map),
            "mean": {key: float(np.mean([r[key] for r in rows])) for key in METRICS},
            "score_ci": interval(per_map),
            "suicides_total": sum(r["suicides"] for r in rows),
            "timeouts_total": sum(r["timeouts"] for r in rows),
            "per_map_score": per_map.tolist(),
            "per_map_coins": [
                float(np.mean([r["coins"] for r in rows if r["seed"] == s]))
                for s in protocol["seeds"]
            ],
        }
    for (name, preset), values in maps.items():
        delta = values - maps[("reference", preset)]
        summaries[name][preset].update(
            delta=float(delta.mean()), delta_ci=interval(delta)
        )
    panels: dict[str, Any] = {}
    for agent in ("dqn_agent", "tabular_q_agent"):
        panels[agent] = {}
        for preset in protocol["presets"]:
            values = np.stack([maps[(f"d8_{agent}_seed{s}", preset)] for s in range(3)])
            rng = np.random.default_rng(20260921)
            outer = rng.integers(3, size=(20000, 3, 1))
            inner = rng.integers(100, size=(20000, 3, 100))
            boots = values[outer, inner].mean(axis=(1, 2))
            deltas = (values[outer, inner] - maps[("reference", preset)][inner]).mean(
                axis=(1, 2)
            )
            panels[agent][preset] = {
                "score": float(values.mean()),
                "score_ci": np.quantile(boots, [0.025, 0.975]).tolist(),
                "delta": float((values - maps[("reference", preset)]).mean()),
                "delta_ci": np.quantile(deltas, [0.025, 0.975]).tolist(),
                "seed_scores": values.mean(axis=1).tolist(),
            }
    hsummary: dict[str, Any] = {}
    for agent in ("dqn_agent", "tabular_q_agent"):
        rows = [next(a for a in r["agents"] if a["code_name"] == agent) for r in head]
        values = np.array(
            [
                np.mean(
                    [
                        a["score"]
                        for r, a in zip(head, rows, strict=True)
                        if r["seed"] == s
                    ]
                )
                for s in protocol["head_to_head"]["seeds"]
            ]
        )
        hsummary[agent] = {
            "games": len(rows),
            "score_ci": interval(values),
            "mean": {k: float(np.mean([r[k] for r in rows])) for k in METRICS},
            "per_map_score": values.tolist(),
        }
    hd = np.array(hsummary["dqn_agent"]["per_map_score"]) - np.array(
        hsummary["tabular_q_agent"]["per_map_score"]
    )
    hsummary["paired_dqn_minus_table"] = {"mean": float(hd.mean()), "ci": interval(hd)}
    winner, reason = choose(
        summaries["dqn_stage2_baseline"]["vs-rule-based"],
        summaries["tabular_blended"]["vs-rule-based"],
    )
    latency: dict[str, Any] = {}
    lcomplete = json.loads((out / "latency/complete.json").read_text())
    assert lcomplete["jobs"] == 1 and lcomplete["games"] == 50
    for candidate in protocol["eligible"]:
        samples: list[float] = []
        timeouts = 0
        matches = 0
        for task in schedule(protocol, "latency"):
            if task["candidate"] != candidate:
                continue
            row = json.loads((out / "latency/games" / f"{task['id']}.json").read_text())
            assert row["status"] == "complete"
            original = json.loads(
                (out / "heldout/games" / f"{task['id']}.json").read_text()
            )
            assert row["initial_board_sha256"] == original["initial_board_sha256"]
            f = row["result"]["agents"][0]
            timeouts += f["timeouts"]
            matches += all(f[k] == original["result"]["agents"][0][k] for k in METRICS)
            samples.extend(row["latencies"][0])
        latency[candidate] = {
            "games": 25,
            "actions": len(samples),
            "timeouts": timeouts,
            "quality_games_matching_gameplay": matches,
            "mean_ms": float(np.mean(samples) * 1000),
            "p99_ms": float(np.quantile(samples, 0.99, method="higher") * 1000),
            "max_ms": max(samples) * 1000,
        }
    return {
        "protocol_sha256": sha(PROTOCOL),
        "evidence_sha256": evidence.hexdigest(),
        "games": completed["games"],
        "seconds": completed["seconds"],
        "summaries": dict(summaries),
        "training_seed_panels": panels,
        "head_to_head": hsummary,
        "latency": latency,
        "selection": {"winner": winner, "reason": reason},
        "limitations": [
            "Practical candidates are single training histories, "
            "not matched-budget learners.",
            "Practical model CIs measure map variability, "
            "not independent training variability.",
            "Reduced D8 omits combat curriculum; D6/D7 gates remain failed.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Do not replace published results")
    write(args.report, summarise(args.out))
