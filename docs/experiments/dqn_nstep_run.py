"""One predeclared 3-step comparison; existing D7 driver, evaluation and stops."""

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from agent_code.dqn_agent.learner import Learner, UpdateStats
from agent_code.dqn_agent.network import QNetwork
from agent_code.dqn_agent.persistence import load_checkpoint
from agent_code.dqn_agent.replay import ReplayBatch
from agent_code.dqn_agent.train import Trainer
from docs.experiments import dqn_d7
from docs.experiments.dqn_d6 import ENCODER, checksum, dump
from docs.experiments.dqn_d7_diagnose import collect
from docs.experiments.dqn_nstep_prepare import equal
from training.config import Curriculum, load_curriculum
from training.continuation import fork_curriculum

CONFIG = Path("docs/experiments/dqn_d10_nstep3.json")
CONTROL = Path("results/dqn/d7_stage2_seed0_20260918")


def run(directory: Path) -> None:
    control, treatment = load_curriculum(dqn_d7.CONFIG), load_curriculum(CONFIG)
    expected = asdict(control)
    expected["name"] = treatment.name
    expected["params"]["n_step"] = 3
    assert expected == asdict(treatment)
    old = json.loads((CONTROL / "summary.json").read_text())
    assert all(
        checksum(dqn_d7.PARENT / n) == sha for n, sha in old["parent_sha256"].items()
    )
    protected = {
        str(p): checksum(p)
        for root in (CONTROL, dqn_d7.PARENT)
        for p in root.rglob("*")
        if p.is_file() and p.suffix in (".npz", ".pt", ".json", ".jsonl")
    }
    original_update, original_evaluate = Learner.update, dqn_d7.evaluate
    recorded = False

    def checked_fork(parent: Path, out: Path, course: Curriculum, seed: int) -> Trainer:
        trainer = fork_curriculum(parent, out, course, seed, convert_n_step=True)
        stored = load_checkpoint(parent / "checkpoint.pt")
        assert equal(stored["learner"], trainer.learner.training_state())
        assert equal(stored["feature_rng"], trainer.agent.rng.getstate())
        assert trainer.transitions == len(trainer.replay) == 50047
        assert trainer.learner.updates == 11262 and trainer.stage_position == 0
        dump(
            out / "comparison_protocol.json",
            {
                "config_difference": {
                    "n_step": [1, 3],
                    "name": [control.name, course.name],
                },
                "primary": "final mean loot coins vs 28.4 on seeds 500-509",
                "stage2_pass": "loot mean >43.05 and zero suicides in 10 rounds",
                "secondary": [
                    "coin-heaven retention",
                    "classic coins",
                    "100k dip",
                    "final three-map bombing/loop diagnostics",
                ],
                "planned_stage_transitions": 300000,
                "training_seed": 0,
                "initial_learner_rng_identical": True,
                "before_sha256": protected,
                "initial_evaluation": "reuse original D6 after checking every weight",
                "stop": "nonfinite Q or abs(Q)>50; no automatic retry",
            },
        )
        shutil.copyfile(Path(__file__), out / "nstep_run_source.py")
        return trainer

    def checked_update(learner: Learner, batch: ReplayBatch) -> UpdateStats:
        nonlocal recorded
        assert learner.config.n_step == 3
        assert all(g["lr"] == 3e-4 for g in learner.optimizer.param_groups)
        assert (batch.k[~batch.done] == 3).all()
        if not recorded:
            dump(
                directory / "first_update.json",
                {
                    "updates_before": learner.updates,
                    "n_step": 3,
                    "lr": 3e-4,
                    "target_every": learner.config.target_every,
                },
            )
            recorded = True
        return original_update(learner, batch)

    def evaluate(model: Path, out: Path) -> list[dict[str, Any]]:
        if model.parent.name != "transition_000050047":
            return original_evaluate(model, out)
        initial = CONTROL / "checkpoints/transition_000050047"
        a, b = (
            QNetwork.load(model, ENCODER),
            QNetwork.load(initial / "q_net.npz", ENCODER),
        )
        for name in a.arrays:
            np.testing.assert_array_equal(a.arrays[name], b.arrays[name])
        shutil.copytree(initial / "evaluation", out)
        return json.loads((out / "summary.json").read_text())

    with (
        patch.object(dqn_d7, "CONFIG", CONFIG),
        patch.object(dqn_d7, "fork_curriculum", checked_fork),
        patch.object(dqn_d7, "evaluate", evaluate),
        patch.object(Learner, "update", checked_update),
    ):
        dqn_d7.run(directory)
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["failure"] is None and recorded
    collect(
        directory / "diagnostics",
        root=directory,
        counts=(summary["global_transitions"],),
        scenarios=("loot-crate",),
    )
    assert all(checksum(Path(p)) == sha for p, sha in protected.items())
    dump(
        directory / "originals_unchanged.json",
        {
            "unchanged": True,
            "verified_files": len(protected),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.out.resolve())


if __name__ == "__main__":
    main()
