"""Fork a finished run into an ``n_step=3`` continuation and audit the copy.

Copies the parent's full state, applies the curriculum's parameter change and
checks the result; it plays no games and performs no updates. The parent is
read-only. Run from the repository root::

    uv run python -m docs.experiments.dqn_nstep_prepare \\
        --parent results/dqn/d6/lr3e-04 --out results/dqn/nstep3 \\
        --config docs/experiments/dqn_d10_nstep3.json
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from agent_code.dqn_agent.persistence import load_checkpoint, load_replay
from training.config import load_curriculum
from training.continuation import fork_curriculum


def equal(a: Any, b: Any) -> bool:  # noqa: ANN401
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return bool(np.array_equal(cast(Any, a), b))
    if isinstance(a, dict):
        obj = cast(dict[str, Any], a)
        return obj.keys() == b.keys() and all(equal(v, b[k]) for k, v in obj.items())
    if isinstance(a, (list, tuple)):
        items = cast(list[Any], a)
        return len(items) == len(b) and all(
            equal(x, y) for x, y in zip(items, b, strict=True)
        )
    return bool(a == b)


def prepare(parent: Path, destination: Path, config: Path) -> None:
    before = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in parent.glob("*")
        if p.suffix in (".pt", ".npz")
    }
    original = load_checkpoint(parent / "checkpoint.pt")
    replay, stale = load_replay(parent / "replay.npz", original, original["schema_id"])
    assert not stale
    source = replay.snapshot()
    trainer = fork_curriculum(
        parent, destination, load_curriculum(config), 0, convert_n_step=True
    )
    child = load_checkpoint(destination / "checkpoint.pt")
    assert equal(original["learner"], child["learner"])
    assert equal(original["feature_rng"], child["feature_rng"])
    assert child["transitions"] == original["transitions"]
    assert child["rounds_trained"] == original["rounds_trained"]
    converted, stale = load_replay(
        destination / "replay.npz", child, child["schema_id"]
    )
    assert not stale and trainer.stage_position == 0 and not trainer.nstep.pending
    result = converted.snapshot()
    assert source["write"] == result["write"] and len(converted) == len(replay)
    for name in ("x", "a", "transition_id", "round_id", "stage"):
        np.testing.assert_array_equal(source["rows"][name], result["rows"][name])
    assert before == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in parent.glob("*")
        if p.suffix in (".pt", ".npz")
    }
    # Round-boundary resume must accept the converted pair with no update.
    # load_replay above checks its generation/IDs; Trainer resume is covered in tests.
    k, counts = np.unique(result["rows"]["k"], return_counts=True)
    report = {
        "parent": str(parent),
        "parent_checksums": before,
        "destination": str(destination),
        "config": str(config),
        "transitions": trainer.transitions,
        "updates": trainer.learner.updates,
        "rounds": trainer.rounds_trained,
        "replay_size": len(converted),
        "write": result["write"],
        "source_terminals": int(source["rows"]["done"].sum()),
        "sequence_lengths": {
            str(int(a)): int(b) for a, b in zip(k, counts, strict=True)
        },
        "learner_and_rng_unchanged": True,
        "parent_unchanged": True,
        "games_run": 0,
        "updates_run": 0,
        "lr": [g["lr"] for g in trainer.learner.optimizer.param_groups],
    }
    (destination / "preparation_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("docs/experiments/dqn_d10_nstep3.json")
    )
    args = parser.parse_args()
    prepare(args.parent, args.out, args.config)


if __name__ == "__main__":
    main()
