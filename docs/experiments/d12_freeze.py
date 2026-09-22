"""Summarize 25 blended validation games and freeze the future protocol; no games."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from docs.experiments.d8_report import paired
from docs.experiments.d12_prepare import sha


def freeze() -> None:
    directory = Path("docs/experiments")
    manifest = json.loads((directory / "d12_manifest.json").read_text())
    old_path = directory / "dqn_candidates_validation.json"
    old = json.loads(old_path.read_text())
    new_path = Path("results/dqn/d12_blended_validation_20260921/results.json")
    new = json.loads(new_path.read_text())
    rows = {
        "blended": new["rows"]["tabular_q_agent"],
        "dqn": old["rows"]["dqn_agent"],
        "reference": old["rows"]["rule_based_agent"],
    }
    reference = rows["reference"]
    metrics: dict[str, Any] = {}
    for name, records in rows.items():
        assert [r["seed"] for r in records] == list(range(500, 525))
        assert [r["initial_board_sha256"] for r in records] == [
            r["initial_board_sha256"] for r in reference
        ]
        metrics[name] = {
            **{
                k: float(np.mean([r[k] for r in records]))
                for k in ("score", "coins", "kills", "steps")
            },
            **{
                k: sum(r[k] for r in records)
                for k in ("suicides", "survived", "invalid", "timeouts")
            },
            "delta_reference": paired(
                [
                    a["score"] - b["score"]
                    for a, b in zip(records, reference, strict=True)
                ]
            ),
        }
    evaluation = {
        "new_games": 25,
        "reused_games": 50,
        "seconds_new": new["seconds"],
        "sources": {str(p): sha(p) for p in (new_path, old_path)},
        "metrics": metrics,
        "rows": rows,
        "blended_minus_dqn": paired(
            [
                a["score"] - b["score"]
                for a, b in zip(rows["blended"], rows["dqn"], strict=True)
            ]
        ),
        "ci_note": "Paired maps,20000 draws,RNG20260921; "
        "not independent-training uncertainty",
    }
    selected: list[dict[str, Any]] = []
    for c in manifest["candidates"]:
        assert sha(Path(c["source"])) == c["sha256"]
        if c["id"] == "tabular_previous":
            continue
        selected.append(
            {
                **c,
                "role": "submission_candidate"
                if c["id"] in ("dqn_stage2_baseline", "tabular_blended")
                else "matched_seed_diagnostic_not_submission_selection",
            }
        )
    protocol = {
        "status": "frozen before held-out; no held-out results collected",
        "eligible": ["dqn_stage2_baseline", "tabular_blended"],
        "candidates": selected,
        "packages": {
            k: {
                field: value[field]
                for field in ("zip", "zip_sha256", "model_sha256", "members", "backup")
            }
            for k, value in manifest["packages"].items()
        },
        "candidate_selection_reason": "Retain previously designated practical DQN; "
        "accept current upstream blended table after compatible validation. "
        "No best D8 seed is selected after seeing held-out results.",
        "presets": {
            "vs-rule-based": {"seats": [0, 1, 2, 3], "primary": True},
            "vs-peaceful": {"seats": [0, 1, 2, 3]},
            "vs-coin-collector": {"seats": [0, 1, 2, 3]},
            "mixed": {"seats": [0, 1, 2, 3]},
            "coin-heaven-solo": {"seats": [0]},
            "crates-solo": {"seats": [0]},
        },
        "seeds": list(range(3000, 3100)),
        "candidate_rng": 500,
        "opponent_rng": "private NumPy RandomState and Python Random: "
        "10*world_seed + seat; setup entropy isolated and caller RNG restored",
        "reference": "rule_based_agent in candidate seat, same opponent streams; "
        "shared only when preset/seed/seat/framework/RNG protocol match exactly",
        "head_to_head": {
            "seeds": list(range(3000, 3050)),
            "lineup": ["frozen_dqn", "frozen_table", "rule_A", "rule_B"],
            "permutations": 24,
            "rule_rng": "10*world_seed+seat+1000000*identity; rule_A=0,rule_B=1",
            "games": 1200,
        },
        "round_limit": 400,
        "train": False,
        "epsilon": 0,
        "inference": "NumPy,strict explicit model",
        "jobs": 2,
        "latency_jobs": 1,
        "budget_games": {
            "per_model": 1800,
            "models": 8,
            "shared_reference": 1800,
            "head_to_head": 1200,
            "total": 17400,
        },
        "statistics": {
            "primary": "mean raw score and paired delta vs reference",
            "interval": "95% percentile bootstrap,20000 draws,RNG20260921; "
            "resample world seeds, retaining all seats/lineup permutations in cluster",
            "matched_study": "also resample training seeds as outer unit; "
            "keep practical baseline and blended model separate from D8 seed panels",
            "selection": "Compare practical candidates only: larger paired delta; "
            "overlapping delta CIs -> fewer suicides in primary games; "
            "remaining tie -> table",
            "additional": [
                "credited kills",
                "coins",
                "suicides",
                "survival",
                "invalid",
                "timeouts",
            ],
        },
        "failures": "Record all failed/limited games; no post-hoc seed removal, "
        "tuning, model replacement or automatic budget extension",
        "integrity": "Verify board/positions, frozen hashes, explicit loaded arrays, "
        "no trainer/Torch, no writes to weights; preserve every per-game row",
        "scope_note": "Q12 unavailable, confirmed by user; secondary presets are "
        "an explicit operational definition from repository tasks, "
        "not claimed Q12 text. "
        "Full D8 combat curriculum is still absent; no training is added.",
    }
    for filename, value in (
        ("d12_blended_validation.json", evaluation),
        ("d12_frozen_protocol.json", protocol),
    ):
        path = directory / filename
        if path.exists():
            raise FileExistsError("frozen artifacts must not be silently overwritten")
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")


if __name__ == "__main__":
    freeze()
