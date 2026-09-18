"""Start a new DQN curriculum from a validated full save, without editing history.

The parent is read-only. Ordinary resume remains strict about config and probe.
This explicit stage boundary permits a new diagnostic probe, but never resets
optimizer, target, replay, RNG or global counters. The new cursor owns only the
new curriculum; the parent's completed chunks are not copied or rewritten.
"""

import hashlib
import json
import logging
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from agent_code.dqn_agent.callbacks import AgentSelf, setup
from agent_code.dqn_agent.config import REPO_ROOT
from agent_code.dqn_agent.persistence import load_checkpoint
from agent_code.dqn_agent.probe import Probe
from agent_code.dqn_agent.train import Trainer
from training.config import Curriculum
from training.dqn import stage_config
from training.driver import environment, start_or_resume


def fork_curriculum(
    parent: Path, destination: Path, course: Curriculum, seed: int
) -> Trainer:
    """Create a fresh experiment directory, returning its fully restored trainer.

    Failure is fail-closed: a nonempty destination is never overwritten. Resume
    an already prepared branch with train_run(course, destination, seed).
    """
    parent, destination = parent.resolve(), destination.resolve()
    if destination == parent or (destination.exists() and any(destination.iterdir())):
        raise FileExistsError("continuation needs a new empty destination")
    if course.agent != "dqn_agent":
        raise ValueError("full continuation supports DQN only")
    checkpoint = parent / "checkpoint.pt"
    stored = load_checkpoint(checkpoint)
    holder = cast(
        AgentSelf, SimpleNamespace(train=True, logger=logging.getLogger(__name__))
    )
    with environment(
        {
            "DQN_AGENT_MODEL": str(parent / "q_net.npz"),
            "DQN_AGENT_PARAMS": json.dumps(stored["config"]),
        }
    ):
        setup(holder)
        trainer = Trainer(holder)
    if not trainer.exact_history:
        raise ValueError("continuation requires a consistent checkpoint/replay pair")
    offset = trainer.config.stage + 1
    config = stage_config(course, course.stages[0], offset, seed)
    # Validate all structural fields before changing the optional probe.
    trainer.configure_stage(replace(config, probe_path=trainer.config.probe_path))
    probe = (
        Probe.load(
            REPO_ROOT / config.probe_path, holder.encoder.schema_id, holder.encoder.dim
        )
        if config.probe_path
        else None
    )
    trainer.config = holder.config = trainer.learner.config = config
    trainer.probe = probe
    trainer.parent_checkpoint = str(checkpoint)
    trainer.run_id = uuid.uuid4().hex
    trainer.driver_state = {
        "next_stage": 0,
        "stage_rounds": 0,
        "stage_offset": offset,
        "records": [],
    }
    holder.model_file = destination / "q_net.npz"
    destination.mkdir(parents=True, exist_ok=True)
    start_or_resume(destination, course, seed, None)
    provenance = {
        "parent_checkpoint": str(checkpoint),
        "parent_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "parent_replay_sha256": hashlib.sha256(
            (parent / "replay.npz").read_bytes()
        ).hexdigest(),
        "parent_config": stored["config"],
        "parent_transitions": trainer.transitions,
        "parent_updates": trainer.learner.updates,
        "parent_rounds": trainer.rounds_trained,
        "stage_offset": offset,
    }
    (destination / "continuation.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    trainer.save(full=True)
    return trainer
