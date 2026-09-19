"""One diagnostic stage3 run; unchanged driver/agent, controlled evaluation RNG."""

import argparse
import importlib
import json
import random
import shutil
from collections.abc import Callable, Generator, Sequence
from contextlib import ExitStack, contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np

from agent_code.dqn_agent import callbacks
from agent_code.dqn_agent.core.world_model import ACTIONS, Observation
from agent_code.dqn_agent.features import Extracted, Extractor
from agent_code.dqn_agent.network import QNetwork
from agent_code.dqn_agent.persistence import load_checkpoint, load_replay
from agent_code.dqn_agent.symmetry import canonical, to_canonical
from agent_code.dqn_agent.train import Trainer
from docs.experiments import dqn_d7
from docs.experiments.dqn_d6 import ENCODER, checksum, dump
from docs.experiments.dqn_nstep_prepare import equal
from tournament import engine
from tournament.engine import WorldConfig, quiet_logging
from tournament.results import RoundResult
from tournament.schedule import PRESETS
from tournament.storage import write_rounds
from training.config import Curriculum
from training.continuation import fork_curriculum
from training.driver import environment
from training.world import TrainingWorld, create_training_world

PARENT = Path("results/dqn/d7_stage2_seed0_20260918/checkpoints/transition_000350346")
CONFIG = Path("docs/experiments/dqn_d7_stage3.json")
ORIGIN = 350346
BATTLE = ("vs-peaceful", "vs-coin-collector")


@contextmanager
def opponent_rngs(seed: int) -> Generator[None]:
    """Sequential evaluation only: private streams around unmodified opponents.

    Their setup() seeds NumPy from entropy. Restore the caller's global state
    after setup and each act, retaining per-agent NumPy and Python RNG streams.
    This fixes their random draws, not their trajectories after policies diverge.
    Training uses the stock opponents unchanged, including entropy seeding.
    """
    streams: dict[int, tuple[np.random.RandomState, random.Random]] = {}
    with ExitStack() as stack:
        for name in ("peaceful_agent", "coin_collector_agent"):
            module = importlib.import_module(f"agent_code.{name}.callbacks")
            original_setup = cast(Callable[..., None], module.setup)
            original_act = cast(Callable[..., str], module.act)

            @wraps(original_setup)
            def setup(
                agent: object, original: Callable[..., None] = original_setup
            ) -> None:
                np_state, py_state = np.random.get_state(), random.getstate()
                streams[id(agent)] = (
                    np.random.RandomState(seed * 10 + len(streams)),
                    random.Random(seed * 10 + len(streams)),
                )
                try:
                    original(agent)
                finally:
                    np.random.set_state(np_state)
                    random.setstate(py_state)

            @wraps(original_act)
            def act(
                agent: object,
                state: dict[str, Any],
                original: Callable[..., str] = original_act,
            ) -> str:
                np_state, py_state = np.random.get_state(), random.getstate()
                nr, pr = streams[id(agent)]
                np.random.set_state(nr.get_state())
                random.setstate(pr.getstate())
                try:
                    return original(agent, state)
                finally:
                    nr.set_state(np.random.get_state())
                    pr.setstate(random.getstate())
                    np.random.set_state(np_state)
                    random.setstate(py_state)

            stack.enter_context(patch.object(module, "setup", setup))
            stack.enter_context(patch.object(module, "act", act))
        yield


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    actions = sum(r["actions"] for r in rows)
    totals = {
        k: sum(r[k] for r in rows)
        for k in (
            "score",
            "kills",
            "coins",
            "crates",
            "suicides",
            "survived",
            "bombs",
            "invalid",
            "timeouts",
            "wait",
            "forced",
            "empty_bombs",
            "empty_contexts",
        )
    }
    return {
        "rounds": n,
        "actions": actions,
        "totals": totals,
        "per_round": {k: v / n for k, v in totals.items()},
        "forced_step_fraction": totals["forced"] / actions,
        "wait_fraction": totals["wait"] / actions,
        "mean_steps": sum(r["steps"] for r in rows) / n,
        "all50_rounds": sum(r["coins"] == 50 for r in rows),
        "per_map": rows,
    }


def evaluate_preset(
    model: Path, directory: Path, preset: str, policy: str, seeds: Sequence[int]
) -> dict[str, Any]:
    model = model.resolve()
    directory = directory.resolve()
    expected = QNetwork.load(model, ENCODER)
    original_setup, original_act = callbacks.setup, callbacks.act
    original_extract = Extractor.extract
    current: list[Extracted] = []
    trace: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    raw: list[RoundResult] = []

    @wraps(original_setup)
    def setup(agent: object) -> None:
        handle = cast(Any, agent)
        original_setup(handle)
        assert not handle.train and handle.trainer is None
        assert handle.model_file.resolve() == model.resolve()
        assert isinstance(handle.q_function, QNetwork)
        assert handle.config.policy == policy and handle.config.mask == "best_tier"
        assert handle.config.n_step == 1
        for k in expected.arrays:
            np.testing.assert_array_equal(
                expected.arrays[k], handle.q_function.arrays[k]
            )

    def extract(extractor: Extractor, obs: Observation) -> Extracted:
        value = original_extract(extractor, obs)
        current[:] = [value]
        return value

    @wraps(original_act)
    def act(agent: object, state: dict[str, Any]) -> str:
        handle = cast(Any, agent)
        action = original_act(handle, state)
        ext = current[0]
        index, symmetry = canonical(ext.features, handle.encoding)
        x = ENCODER.encode(handle.encoding.decode(index), ext)
        allowed = [ACTIONS.index(to_canonical(a, symmetry)) for a in ext.allowed]
        trace.append(
            {
                "action": action,
                "forced": len(ext.allowed) == 1,
                "empty": ext.features.bomb_yield == 0 and ext.features.attack == 0,
                "score": state["self"][1],
                "max_q": float(expected.values(x)[allowed].max()),
                "phi": 0
                if ext.coin_distance is None
                else 0.5 / (1 + ext.coin_distance),
            }
        )
        return action

    def world(config: WorldConfig, log_dir: str = "logs") -> TrainingWorld:
        return create_training_world(
            config.lineup, config.scenario, config.seed, train_seats=0, log_dir=log_dir
        )

    for seed in seeds:
        trace.clear()
        with (
            opponent_rngs(seed),
            quiet_logging(),
            environment(
                {
                    "DQN_AGENT_MODEL": str(model.resolve()),
                    "DQN_AGENT_PARAMS": json.dumps({"seed": 500, "policy": policy}),
                }
            ),
            patch.object(callbacks, "setup", setup),
            patch.object(callbacks, "act", act),
            patch.object(Extractor, "extract", extract),
            patch.object(engine, "create_world", world),
        ):
            target = PRESETS[preset]
            result = engine.run_round(
                WorldConfig(("dqn_agent", *target.opponents), target.scenario, seed),
                log_dir=str(directory / "logs" / f"{preset}_{seed}"),
            )
        raw.append(result)
        focus = result.focus
        returns = dqn_d7.discounted_returns(
            [r["score"] for r in trace], [r["phi"] for r in trace], focus.score, 0.99
        )
        gaps = [r["max_q"] - value for r, value in zip(trace, returns, strict=True)]
        rows.append(
            {
                "seed": seed,
                **{
                    k: getattr(focus, k)
                    for k in (
                        "score",
                        "kills",
                        "coins",
                        "crates",
                        "suicides",
                        "survived",
                        "bombs",
                        "invalid",
                        "timeouts",
                        "steps",
                    )
                },
                "actions": len(trace),
                "wait": sum(r["action"] == "WAIT" for r in trace),
                "forced": sum(r["forced"] for r in trace),
                "empty_bombs": sum(r["empty"] and r["action"] == "BOMB" for r in trace),
                "empty_contexts": sum(r["empty"] for r in trace),
                "mean_q_minus_realized_return": float(np.mean(gaps)),
            }
        )
        dump(directory / f"{preset}_{seed}_decisions.json", trace)
    write_rounds(directory / f"{preset}.jsonl", raw)
    summary = {"preset": preset, "policy": policy, **summarize(rows)}
    dump(directory / f"{preset}_summary.json", summary)
    return summary


def run(directory: Path) -> None:
    original_fork = fork_curriculum
    protected = {str(p): checksum(p) for p in PARENT.rglob("*") if p.is_file()}

    def fork(parent: Path, out: Path, course: Curriculum, seed: int) -> Trainer:
        trainer = original_fork(parent, out, course, seed)
        stored = load_checkpoint(parent / "checkpoint.pt")
        assert stored["config"].get("n_step", 1) == trainer.config.n_step == 1
        assert trainer.config.lr == 3e-4 and trainer.config.target_every == 1000
        assert all(g["lr"] == 3e-4 for g in trainer.learner.optimizer.param_groups)
        assert equal(stored["learner"], trainer.learner.training_state())
        assert equal(stored["feature_rng"], trainer.agent.rng.getstate())
        replay, stale = load_replay(parent / "replay.npz", stored, stored["schema_id"])
        assert not stale and equal(replay.snapshot(), trainer.replay.snapshot())
        assert trainer.transitions == ORIGIN and trainer.stage_position == 0
        dump(
            out / "protocol.json",
            {
                "deviation": (
                    "stage2 quality gate failed; user authorized one diagnostic stage3"
                ),
                "parent_files": protected,
                "parent_transitions": ORIGIN,
                "parent_updates": trainer.learner.updates,
                "optimizer_lr": 3e-4,
                "n_step": 1,
                "evaluation_seeds": list(range(500, 525)),
                "seat": 0,
                "opponent_rng": (
                    "private numpy/Python streams seed=world_seed*10+opponent_index; "
                    "evaluation only"
                ),
                "earlier_tasks": (
                    "8% loot solo, 8% classic solo, 4% coin-heaven solo per block"
                ),
                "epsilon": (
                    "continuous nominal .3 -> .05 over first 240k hunting transitions"
                ),
            },
        )
        shutil.copyfile(Path(__file__), out / "hunting_source.py")
        return trainer

    def evaluate(model: Path, out: Path) -> list[dict[str, Any]]:
        count = int(model.parent.name.split("_")[-1])
        results = [
            evaluate_preset(model, out, p, "learned", range(500, 525)) for p in BATTLE
        ]
        if count == ORIGIN:
            controls = [
                evaluate_preset(
                    model, directory / "safe_random", p, "random", range(500, 525)
                )
                for p in BATTLE
            ]
            dump(directory / "safe_random/summary.json", controls)
            # Parent solo assessments used the same ten maps and inference setup.
            old = json.loads((PARENT / "evaluation/summary.json").read_text())
            for row in old:
                if row["preset"] in ("crates-solo", "coin-heaven-solo"):
                    results.append({**row, "reused_from": str(PARENT / "evaluation")})
        elif count >= ORIGIN + 400000:
            results.extend(
                evaluate_preset(model, out, p, "learned", range(500, 510))
                for p in ("crates-solo", "coin-heaven-solo")
            )
        dump(out / "summary.json", results)
        return results

    with (
        patch.object(dqn_d7, "PARENT", PARENT),
        patch.object(dqn_d7, "CONFIG", CONFIG),
        patch.object(dqn_d7, "fork_curriculum", fork),
        patch.object(dqn_d7, "evaluate", evaluate),
    ):
        dqn_d7.run(directory)
    assert all(checksum(Path(p)) == sha for p, sha in protected.items())
    dump(
        directory / "originals_unchanged.json",
        {"files": len(protected), "unchanged": True},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.out.resolve())


if __name__ == "__main__":
    main()
