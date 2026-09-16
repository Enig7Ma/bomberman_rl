"""Opt-in D2 tracing of actual decisions and execution; no extra RNG draws."""

import hashlib
import json
import os
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
from numpy.typing import NDArray

import events
from agent_code.dqn_agent import callbacks
from agent_code.dqn_agent.config import ENV_VAR, MODEL_ENV_VAR, Config
from agent_code.dqn_agent.core.world_model import ACTIONS, Observation
from agent_code.dqn_agent.features import Extracted, Extractor, Features
from agent_code.dqn_agent.symmetry import canonical, from_canonical, to_canonical
from tournament.engine import create_world, quiet_logging, reset_framework_logging
from tournament.schedule import PRESETS, build_schedule


def json_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return cast(Any, value).item()
    raise TypeError(f"unsupported trace value: {type(value)}")


def snapshot(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in state.items()
    }


def classify(path: Path) -> dict[str, Any]:
    """Count only conflicts whose earlier move is present in the trace."""
    counts: Counter[str] = Counter()
    conflicts = 0
    offsets = {"UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1), "LEFT": (-1, 0)}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        action = row["executed_action"]
        counts[action] += 1
        if action not in offsets:
            continue
        obs, current = row["observation"], row["execution_state"]
        x, y = obs["self"][3]
        dx, dy = offsets[action]
        target = [x + dx, y + dy]
        free_before = (
            obs["field"][target[0]][target[1]] == 0
            and not any(p[3] == target for p in obs["others"])
            and not any(b[0] == target for b in obs["bombs"])
        )
        occupants = {p[0] for p in current["others"] if p[3] == target}
        witnessed = any(
            move["player"] in occupants
            and any(
                p[0] == move["player"] and p[3] != target
                for p in move["before"]["others"]
            )
            and any(
                p[0] == move["player"] and p[3] == target
                for p in move["after"]["others"]
            )
            for move in row["earlier_actions"]
        )
        conflicts += bool(free_before and occupants and witnessed)
    return {
        "invalid_by_action": dict(counts),
        "witnessed_occupancy_conflicts": conflicts,
        "unclassified": sum(counts.values()) - conflicts,
    }


def diagnose(output: Path, model: Path, seed: int) -> None:
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    os.environ[MODEL_ENV_VAR] = str(model)
    os.environ[ENV_VAR] = json.dumps(asdict(Config(seed=0, init_seed=0)))
    actions: Counter[str] = Counter()
    decisions = invalid = 0
    original_act = callbacks.act
    original_extract = Extractor.extract
    original_greedy = callbacks.masked_greedy
    trace: dict[str, Any] = {}

    def extract(self: Extractor, obs: Observation) -> Extracted:
        result = original_extract(self, obs)
        trace["features"] = asdict(result.features)
        trace["original_mask"] = list(result.allowed)
        return result

    def greedy(
        q: NDArray[np.float32], allowed: Sequence[int], rng: random.Random
    ) -> int:
        selected = original_greedy(q, allowed, rng)
        trace["canonical_mask"] = [ACTIONS[i] for i in allowed]
        trace["canonical_action"] = ACTIONS[selected]
        trace["q_values"] = q.tolist()
        return selected

    def act(self: callbacks.AgentSelf, state: Mapping[str, Any]) -> callbacks.Action:
        nonlocal decisions
        trace.clear()
        trace["observation"] = snapshot(state)
        action = original_act(self, state)
        # Recompute the pure symmetry operation, never extract or select twice.
        features = Features(**trace["features"])
        _, symmetry = canonical(features, self.encoding)
        trace["symmetry"] = asdict(symmetry)
        assert set(trace["canonical_mask"]) == {
            to_canonical(a, symmetry) for a in trace["original_mask"]
        }
        assert from_canonical(trace["canonical_action"], symmetry) == action
        trace["action"] = action
        decisions += 1
        actions[action] += 1
        assert action in trace["original_mask"]
        return action

    def perform(agent: object, action: str) -> None:
        nonlocal invalid
        before = snapshot(world.get_state_for_agent(focus))
        original_perform(agent, action)
        if agent is focus and events.INVALID_ACTION in focus.events:
            invalid += 1
            file.write(
                json.dumps(
                    {
                        "seed": config.seed,
                        "seat": config.focus_seat,
                        "step": world.step,
                        **trace,
                        "executed_action": action,
                        "execution_state": before,
                        "earlier_actions": list(earlier),
                    },
                    default=json_scalar,
                )
                + "\n"
            )
        earlier.append(
            {
                "player": cast(Any, agent).name,
                "action": action,
                "before": before,
                "after": snapshot(world.get_state_for_agent(focus)),
            }
        )

    schedule = build_schedule(
        "dqn_agent", PRESETS["vs-rule-based"], [seed], with_control=False
    )
    with (output / "invalid_traces.jsonl").open("w", encoding="utf-8") as file:
        with (
            quiet_logging(),
            patch.object(Extractor, "extract", extract),
            patch.object(callbacks, "masked_greedy", greedy),
            patch.object(callbacks, "act", act),
        ):
            for config in schedule:
                world: Any = create_world(config, str(output / "logs"))
                try:
                    focus = world.agents[config.focus_seat]
                    assert focus.backend.runner.fake_self.q_function is not None
                    original_perform = world.perform_agent_action
                    earlier: list[dict[str, Any]] = []

                    world.perform_agent_action = perform
                    world.new_round()
                    while world.running:
                        earlier.clear()
                        world.do_step()
                finally:
                    reset_framework_logging()
    assert hashlib.sha256(model.read_bytes()).hexdigest() == digest
    summary = {
        "seed": seed,
        "rounds": 4,
        "decisions": decisions,
        "actions": dict(actions),
        "invalid": invalid,
        "model_sha256": digest,
        **classify(output / "invalid_traces.jsonl"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
