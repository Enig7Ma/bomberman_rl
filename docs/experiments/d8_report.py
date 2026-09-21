"""Rebuild reduced-D8 tables from completed raw results; never plays games."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from docs.experiments.dqn_d6 import checksum, dump


def paired(values: list[float]) -> dict[str, Any]:
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20260921)
    samples = a[rng.integers(0, len(a), (20000, len(a)))].mean(axis=1)
    return {
        "mean": float(a.mean()),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
    }


def hierarchical(values: list[list[float]]) -> dict[str, Any]:
    """Resample training seeds, then games within each selected seed."""
    a = np.asarray(values, dtype=float)
    if a.ndim != 2 or min(a.shape) == 0:
        raise ValueError("nonempty rectangular seed-by-game results required")
    rng = np.random.default_rng(20260921)
    seeds = rng.integers(0, a.shape[0], (20000, a.shape[0], 1))
    games = rng.integers(0, a.shape[1], (20000, *a.shape))
    means = a[seeds, games].mean(axis=(1, 2))
    return {
        "mean": float(a.mean()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "training_seeds": a.shape[0],
        "games_per_seed": a.shape[1],
    }


def report(root: Path, out: Path) -> None:
    study = json.loads((root / "study.json").read_text())
    reference = json.loads(
        Path("docs/experiments/dqn_candidates_validation.json").read_text()
    )
    refs = {r["seed"]: r for r in reference["rows"]["rule_based_agent"]}
    random_rows = study["safe_random"]["rows"]
    random_score = float(
        np.mean([r["score"] for r in random_rows if r["preset"] == "vs-rule-based"])
    )
    curves: list[dict[str, Any]] = []
    boards: dict[tuple[str, int], str] = {}
    for run in study["runs"]:
        for milestone, point in enumerate(run["evaluations"]):
            for r in point["rows"]:
                k = r["preset"], r["seed"]
                assert boards.setdefault(k, r["board"]) == r["board"]
                if r["preset"] == "vs-rule-based":
                    assert r["board"] == refs[r["seed"]]["initial_board_sha256"]
            grouped: dict[str, Any] = {}
            for preset in ("coin-heaven-solo", "crates-solo", "vs-rule-based"):
                rs = [r for r in point["rows"] if r["preset"] == preset]
                means = {
                    k: float(np.mean([r[k] for r in rs]))
                    for k in ("score", "coins", "kills", "crates", "bombs", "steps")
                }
                wins = [r for r in rs if r["coins"] == 50]
                grouped[preset] = {
                    **means,
                    "score_map_ci": paired([r["score"] for r in rs]),
                    "all50": len(wins),
                    "rounds": len(rs),
                    "successful_steps": float(np.mean([r["steps"] for r in wins]))
                    if wins
                    else None,
                    "suicides": sum(r["suicides"] for r in rs),
                    "survived": sum(r["survived"] for r in rs),
                    "invalid": sum(r["invalid"] for r in rs),
                    "timeouts": sum(r["timeouts"] for r in rs),
                    "per_map": {
                        key: [r[key] for r in rs]
                        for key in (
                            "seed",
                            "score",
                            "coins",
                            "kills",
                            "steps",
                            "suicides",
                            "survived",
                            "crates",
                            "bombs",
                            "invalid",
                            "timeouts",
                        )
                    },
                    "delta_reference": paired(
                        [r["score"] - refs[r["seed"]]["score"] for r in rs]
                    )
                    if preset == "vs-rule-based"
                    else None,
                }
            curves.append(
                {
                    "agent": run["agent"],
                    "seed": run["seed"],
                    "milestone": milestone,
                    "transitions": point["transitions"],
                    "updates": point["updates"],
                    "seconds": point["seconds"],
                    "checkpoint": point["directory"],
                    "hashes": point["hashes"],
                    "probe": point["probe"],
                    "visit_bins": point["visit_bins"],
                    "metrics": grouped,
                }
            )
    for r in random_rows:
        assert r["board"] == boards[r["preset"], r["seed"]]
    comparisons: list[dict[str, Any]] = []
    for seed in (0, 1, 2):
        arms = [r for r in study["runs"] if r["seed"] == seed]
        if len(arms) != 2:
            continue
        a, b = arms
        by_name = {r["agent"]: r for r in (a, b)}
        for x, y in zip(
            by_name["dqn_agent"]["evaluations"],
            by_name["tabular_q_agent"]["evaluations"],
            strict=False,
        ):
            dx = {
                r["seed"]: r["score"]
                for r in x["rows"]
                if r["preset"] == "vs-rule-based"
            }
            ty = {
                r["seed"]: r["score"]
                for r in y["rows"]
                if r["preset"] == "vs-rule-based"
            }
            comparisons.append(
                {
                    "seed": seed,
                    "dqn_transitions": x["transitions"],
                    "table_transitions": y["transitions"],
                    "dqn_minus_table": paired([dx[s] - ty[s] for s in sorted(dx)]),
                }
            )
    summary: dict[str, Any] = {
        "analysis_sha256": checksum(Path(__file__)),
        "study": str(root),
        "environment": json.loads((root / "environment.json").read_text()),
        "wall_seconds": study["seconds"],
        "safe_random_score": random_score,
        "safe_random_rows": random_rows,
        "curves": curves,
        "paired_comparisons": comparisons,
        "runs": [
            {k: v for k, v in r.items() if k not in ("points", "evaluations")}
            for r in study["runs"]
        ],
    }
    diagnostics: list[dict[str, Any]] = []
    for run in study["runs"]:
        folder = root / run["agent"] / f"run_{run['seed']}"
        chunks = [
            json.loads(x) for x in (folder / "chunks.jsonl").read_text().splitlines()
        ]
        metrics = [
            json.loads(x) for x in (folder / "metrics.jsonl").read_text().splitlines()
        ]
        mix: dict[str, dict[str, int]] = {}
        for chunk in chunks:
            key = chunk["stage"] + ":" + chunk["scenario"]
            row = mix.setdefault(key, {"rounds": 0, "transitions": 0})
            row["rounds"] += chunk["rounds"]
            row["transitions"] += chunk["transitions"]
        probes = [p["probe"]["max_abs_q"] for p in run["points"]]
        probes.extend(p["max_abs_q"] for m in metrics for p in m.get("probe", []))
        diagnostics.append(
            {
                "agent": run["agent"],
                "seed": run["seed"],
                "actual_mix": mix,
                "max_probe_abs_q": max(probes),
                "chunk_seconds": sum(c["wall_time"] for c in chunks),
                "longest_chunks": [
                    {k: c[k] for k in ("index", "transitions", "wall_time")}
                    for c in sorted(chunks, key=lambda c: c["wall_time"], reverse=True)[
                        :3
                    ]
                ],
                "first_round": metrics[0],
                "last_round": metrics[-1],
            }
        )
    summary["training_diagnostics"] = diagnostics
    final_scores: dict[str, list[list[float]]] = {}
    for agent in ("tabular_q_agent", "dqn_agent"):
        final_scores[agent] = [
            [
                row["score"]
                for row in run["evaluations"][-1]["rows"]
                if row["preset"] == "vs-rule-based"
            ]
            for run in sorted(study["runs"], key=lambda r: r["seed"])
            if run["agent"] == agent and run["evaluations"]
        ]
    summary["hierarchical_final_combat"] = {
        agent: hierarchical(values) for agent, values in final_scores.items()
    }
    summary["hierarchical_final_combat"]["dqn_minus_table"] = hierarchical(
        (
            np.asarray(final_scores["dqn_agent"])
            - np.asarray(final_scores["tabular_q_agent"])
        ).tolist()
    )
    out.mkdir(parents=True, exist_ok=True)
    dump(out / "d8_reduced_results.json", summary)
    lines = [
        "# Reduced D8 results",
        "",
        "Training/evaluation protocol: "
        "[d8_reduced_protocol.md](d8_reduced_protocol.md).",
        "D6/D7 are not reclassified. This excludes hunting/combat training and "
        "all-seat head-to-head; full D8/D12 are not complete.",
        "",
        "## Learning curves",
        "",
        "| Agent | Seed | Actual transitions | Updates | Train seconds | Combat "
        "score [map95% CI] | Loot coins | Navigation coins / all50 |",
        "|---|---:|---:|---:|---:|---|---:|---|",
    ]
    for c in curves:
        metrics = c["metrics"]
        battle = metrics["vs-rule-based"]
        ci = battle["score_map_ci"]["ci95"]
        nav = metrics["coin-heaven-solo"]
        lines.append(
            f"| {c['agent']} | {c['seed']} | {c['transitions']} | {c['updates']} | "
            f"{c['seconds']:.1f} | {battle['score']:.2f} [{ci[0]:.2f},{ci[1]:.2f}] |"
            f" {metrics['crates-solo']['coins']:.2f} | {nav['coins']:.2f} / "
            f"{nav['all50']}/{nav['rounds']} |"
        )
    lines += [
        "",
        "Times include driver saves/archive/probe and worker contention, exclude "
        "evaluation and process import startup. Milestones are actual chunk "
        "boundaries, not invented exact budgets.",
        "",
        "## Final training-seed spread",
        "",
        "| Agent | Mean combat | Min / max | Mean loot | Mean navigation | First "
        "sampled random+1 crossing (transitions per seed) |",
        "|---|---:|---|---:|---:|---|",
    ]
    for agent in ("tabular_q_agent", "dqn_agent"):
        finals: list[dict[str, Any]] = []
        crossings: list[int | None] = []
        for seed in (0, 1, 2):
            cs = [c for c in curves if c["agent"] == agent and c["seed"] == seed]
            if cs:
                finals.append(cs[-1])
                crossings.append(
                    next(
                        (
                            c["transitions"]
                            for c in cs
                            if c["metrics"]["vs-rule-based"]["score"]
                            >= random_score + 1
                        ),
                        None,
                    )
                )
        scores = [c["metrics"]["vs-rule-based"]["score"] for c in finals]
        loot = np.mean([c["metrics"]["crates-solo"]["coins"] for c in finals])
        nav = np.mean([c["metrics"]["coin-heaven-solo"]["coins"] for c in finals])
        lines.append(
            f"| {agent} | {np.mean(scores):.3f} | {min(scores):.2f} / "
            f"{max(scores):.2f} | {loot:.3f} | {nav:.3f} | {crossings} |"
        )
    lines += [
        "",
        f"Safe-random combat reference mean={random_score:.2f}; crossing threshold="
        f"{random_score + 1:.2f}. None means not reached at a sampled point; no "
        f"interpolation.",
        "",
        "Intervals are paired/map bootstrap,20000 resamples,RNG20260921, not "
        "variability of independent trainings. Training-seed spread is reported "
        "separately; three seeds remain a small sample. Raw per-map results are "
        "retained under each exact archive's evaluation directory. JSON links every"
        " curve point to its archived model hashes.",
        "",
        "## Paired direct combat comparison",
        "",
        "| Seed | DQN transitions | Table transitions | DQN minus table [map95% CI] |",
        "|---|---:|---:|---|",
    ]
    for c in comparisons:
        p = c["dqn_minus_table"]
        lo, hi = p["ci95"]
        lines.append(
            f"| {c['seed']} | {c['dqn_transitions']} | {c['table_transitions']} | "
            f"{p['mean']:.2f} [{lo:.2f},{hi:.2f}] |"
        )
    lines += [
        "",
        "Final hierarchical bootstrap resamples training seeds first, then maps "
        "within each seed (20000 draws, RNG20260921). For the direct difference "
        "each map/seed pair remains paired. Three outer units give limited "
        "evidence of training robustness:",
        "",
    ]
    for name, value in summary["hierarchical_final_combat"].items():
        lo, hi = value["ci95"]
        lines.append(f"- {name}: {value['mean']:.3f} [{lo:.3f}, {hi:.3f}].")
    lines += [
        "",
        "## Visit bins at final snapshots",
        "",
        "| Agent | Seed | Table visit bin | Decisions | Table-greedy membership | "
        "Raw score attributed |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for run in study["runs"]:
        if not run["evaluations"]:
            continue
        point = run["evaluations"][-1]
        for b, v in point["visit_bins"].items():
            fraction = (
                v["greedy_agreement"] / v["decisions"] if v["decisions"] else None
            )
            lines.append(
                f"| {run['agent']} | {run['seed']} | {b} | {v['decisions']} | "
                f"{fraction} | {v['base_score']} |"
            )
    lines += [
        "",
        "These are policy-dependent visited states pooled over the three validation"
        " tasks; score attribution is the next observed score increment (terminal "
        "residual includes posthumous kills). Membership accepts ties in the table "
        "greedy set. Initial unvisited ties can inflate agreement; these bins do "
        "not prove causal generalization.",
        "",
        "## Run integrity and timing",
        "",
    ]
    for r in summary["runs"]:
        lines.append(
            f"- {r['agent']} seed{r['seed']}: {r['transitions']} transitions, "
            f"{r['updates']} updates, {r['seconds']:.1f}s, "
            f"{r['transitions'] / r['seconds']:.2f} transitions/s; failure="
            f"{r['failure']!r}."
        )
    lines += [
        "",
        f"Overall elapsed including worker startup/evaluation: "
        f"{study['seconds']:.1f}s. Fixed probe checks and any failed seeds remain "
        f"recorded. Do not equate passing |Q|<=50 with task quality.",
        "",
    ]
    lines += [
        "## Actual training coverage",
        "",
        "| Agent | Seed | Stage:scenario | Rounds | Transitions | Run max abs Q |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for d in diagnostics:
        for task, counts in d["actual_mix"].items():
            lines.append(
                f"| {d['agent']} | {d['seed']} | {task} | "
                f"{counts['rounds']} | {counts['transitions']} | "
                f"{d['max_probe_abs_q']:.3f} |"
            )
    lines += [
        "",
        "## Final available checkpoint details",
        "",
        "Stopped runs show last available models, not completed-budget results.",
        "",
        "| Agent | Seed | Delta ref | Kills | Coins | Suicides/25 | "
        "Survived/25 | Invalid | Timeouts |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in study["runs"]:
        cs = [
            c for c in curves if c["agent"] == run["agent"] and c["seed"] == run["seed"]
        ]
        if not cs:
            continue
        b = cs[-1]["metrics"]["vs-rule-based"]
        lines.append(
            f"| {run['agent']} | {run['seed']} | {b['delta_reference']['mean']:.2f} | "
            f"{b['kills']:.2f} | {b['coins']:.2f} | {b['suicides']} | "
            f"{b['survived']} | {b['invalid']} | {b['timeouts']} |"
        )
    lines += [
        "",
        "All rates use all scheduled rounds, including deaths and limits. "
        "The JSON retains each map and all-round/successful-only steps separately.",
        "",
    ]
    (out / "d8_reduced_results.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("docs/experiments"))
    args = parser.parse_args()
    report(args.root, args.out)
