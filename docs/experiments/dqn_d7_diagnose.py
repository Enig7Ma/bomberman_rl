"""Read-only D7 audit: 24 diagnostic games, no learner or training callbacks.

Four archived policies, loot-crate/classic solo, paired maps 500..502.
The collector observes actual extraction once, so it consumes no agent RNG.
Engine blast attribution is recorded outside the policy, never passed to it.
"""

import argparse
import json
from collections import Counter, deque
from functools import wraps
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
from numpy.typing import NDArray

from agent_code.dqn_agent import callbacks
from agent_code.dqn_agent.core.world_model import ACTIONS, Observation
from agent_code.dqn_agent.features import Extracted, Extractor
from agent_code.dqn_agent.network import QNetwork
from agent_code.dqn_agent.symmetry import canonical, to_canonical
from docs.experiments.dqn_d6 import ENCODER, checksum, dump
from items import Bomb
from tournament.engine import quiet_logging, reset_framework_logging
from training.driver import environment
from training.world import create_training_world

ROOT = Path("results/dqn/d7_stage2_seed0_20260918")
COUNTS = (50047, 150617, 250654, 350346)


def reachable_coins(obs: Observation) -> int:
    """Visible coins reachable on currently free tiles, treating bombs as blocked.

    This is spatial reachability, not a promise of a temporally safe path.
    """
    blocked = {pos for pos, _ in obs.bombs}
    seen = {obs.me.pos}
    queue = deque(seen)
    while queue:
        x, y = queue.popleft()
        for pos in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if (
                0 <= pos[0] < obs.field.shape[0]
                and 0 <= pos[1] < obs.field.shape[1]
                and obs.field[pos] == 0
                and pos not in blocked
                and pos not in seen
            ):
                seen.add(pos)
                queue.append(pos)
    return len(set(obs.coins) & seen)


def summarize(
    trace: list[dict[str, Any]], explosions: list[dict[str, Any]]
) -> dict[str, Any]:
    productive = [
        r for r in trace if "BOMB" in r["allowed"] and r["features"]["bomb_yield"] > 0
    ]
    empty = [
        r
        for r in trace
        if r["features"]["bomb_yield"] == 0 and r["features"]["attack"] == 0
    ]
    bombs = sum(r["action"] == "BOMB" for r in trace)
    return {
        "actions": len(trace),
        "bomb_allowed": sum("BOMB" in r["allowed"] for r in trace),
        "productive_bomb_opportunities": len(productive),
        "productive_bomb_choices": sum(r["action"] == "BOMB" for r in productive),
        "empty_contexts": len(empty),
        "empty_bombs": sum(r["action"] == "BOMB" for r in empty),
        "bombs": bombs,
        "exploded_bombs": len(explosions),
        "crate_destroying_bombs": sum(r["crates"] > 0 for r in explosions),
        "crates": sum(r["crates"] for r in explosions),
        "pending_bombs": bombs - len(explosions),
        "wait": sum(r["action"] == "WAIT" for r in trace),
        "immediate_backtracks": sum(
            trace[i]["pos"] == trace[i - 2]["pos"] != trace[i - 1]["pos"]
            for i in range(2, len(trace))
        ),
        "reachable_visible_coin_steps": sum(r["reachable_coins"] > 0 for r in trace),
        "last_reachable_coins_before_action": trace[-1]["reachable_coins"],
        "score": trace[-1]["score_after"],
        "limit400": len(trace) == 400,
        "actions_histogram": dict(Counter(r["action"] for r in trace)),
    }


def collect(
    directory: Path,
    *,
    root: Path = ROOT,
    counts: tuple[int, ...] = COUNTS,
    scenarios: tuple[str, ...] = ("loot-crate", "classic"),
) -> None:
    if directory.exists():
        raise FileExistsError("use a new directory; never overwrite diagnostic games")
    models = {
        str(n): QNetwork.load(
            root / "checkpoints" / f"transition_{n:09d}" / "q_net.npz", ENCODER
        )
        for n in counts
    }
    protected = [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix in (".npz", ".pt", ".json", ".jsonl")
    ]
    hashes = {str(p): checksum(p) for p in protected}
    directory.mkdir(parents=True)
    dump(
        directory / "manifest.json",
        {
            "round_cap": len(counts) * len(scenarios) * 3,
            "seeds": [500, 501, 502],
            "counts": counts,
            "scenarios": scenarios,
            "root": str(root),
            "before_sha256": hashes,
        },
    )
    original_act, original_extract = callbacks.act, Extractor.extract
    original_blast = cast(Any, Bomb).get_blast_coords
    summaries: list[dict[str, Any]] = []
    for count in counts:
        for scenario in scenarios:
            for seed in (500, 501, 502):
                trace: list[dict[str, Any]] = []
                explosions: list[dict[str, Any]] = []
                extracted_now: list[Extracted] = []

                def capture_extract(
                    extractor: Extractor,
                    obs: Observation,
                    extracted_now: list[Extracted] = extracted_now,
                ) -> Extracted:
                    value = original_extract(extractor, obs)
                    extracted_now[:] = [value]
                    return value

                @wraps(original_act)
                def capture_act(
                    agent: object,
                    state: dict[str, Any],
                    extracted_now: list[Extracted] = extracted_now,
                    trace: list[dict[str, Any]] = trace,
                ) -> str:
                    handle = cast(Any, agent)
                    action = original_act(handle, state)
                    assert not handle.train and handle.trainer is None
                    obs = Observation.from_game_state(state)
                    extracted = extracted_now[0]
                    index, symmetry = canonical(extracted.features, handle.encoding)
                    x = ENCODER.encode(handle.encoding.decode(index), extracted)
                    allowed = extracted.allowed
                    indices = [
                        ACTIONS.index(to_canonical(a, symmetry)) for a in allowed
                    ]
                    comparison = {
                        name: {
                            a: float(net.values(x)[i])
                            for a, i in zip(allowed, indices, strict=True)
                        }
                        for name, net in models.items()
                    }
                    trace.append(
                        {
                            "step": obs.step,
                            "pos": obs.me.pos,
                            "score": state["self"][1],
                            "features": extracted.features.values(),
                            "allowed": allowed,
                            "x": x.tolist(),
                            "q_by_snapshot": comparison,
                            "action": action,
                            "visible_coins": obs.coins,
                            "reachable_coins": reachable_coins(obs),
                            "coin_distance": extracted.coin_distance,
                            "field": obs.field.tolist(),
                            "bombs": obs.bombs,
                            "explosion_map": obs.explosion_map.tolist(),
                        }
                    )
                    return action

                def capture_blast(
                    bomb: object,
                    arena: NDArray[np.int64],
                    explosions: list[dict[str, Any]] = explosions,
                    trace: list[dict[str, Any]] = trace,
                ) -> list[tuple[int, int]]:
                    coords = cast(list[tuple[int, int]], original_blast(bomb, arena))
                    explosions.append(
                        {
                            "step": trace[-1]["step"],
                            "pos": [int(cast(Any, bomb).x), int(cast(Any, bomb).y)],
                            "crates": sum(int(arena[pos] == 1) for pos in coords),
                        }
                    )
                    return coords

                model = root / "checkpoints" / f"transition_{count:09d}" / "q_net.npz"
                label = f"{count}_{scenario}_{seed}"
                with (
                    environment(
                        {
                            "DQN_AGENT_MODEL": str(model.resolve()),
                            "DQN_AGENT_PARAMS": json.dumps(
                                {
                                    "seed": 500,
                                    "policy": "learned",
                                    "encoder": "onehot_e3",
                                }
                            ),
                        }
                    ),
                    quiet_logging(),
                    patch.object(callbacks, "act", capture_act),
                    patch.object(Extractor, "extract", capture_extract),
                    patch.object(Bomb, "get_blast_coords", capture_blast),
                ):
                    try:
                        world = cast(
                            Any,
                            create_training_world(
                                ("dqn_agent",),
                                scenario,
                                seed,
                                train_seats=0,
                                log_dir=str(directory / "logs" / label),
                            ),
                        )
                        handle = world.agents[0].backend.runner.fake_self
                        assert handle.model_file.resolve() == model.resolve()
                        assert isinstance(handle.q_function, QNetwork)
                        for key, value in models[str(count)].arrays.items():
                            np.testing.assert_array_equal(
                                value, handle.q_function.arrays[key]
                            )
                        world.new_round()
                        while world.running:
                            world.do_step()
                            trace[-1]["score_after"] = int(world.agents[0].score)
                            trace[-1]["pos_after"] = [
                                int(world.agents[0].x),
                                int(world.agents[0].y),
                            ]
                            trace[-1]["events"] = list(world.agents[0].events)
                        summary = {
                            "global_transitions": count,
                            "stage_transitions": count - COUNTS[0],
                            "scenario": scenario,
                            "seed": seed,
                            **summarize(trace, explosions),
                        }
                        summary["death"] = bool(world.agents[0].dead)
                        preset = (
                            "crates-solo"
                            if scenario == "loot-crate"
                            else "classic-solo"
                        )
                        old = [
                            json.loads(line)
                            for line in (
                                model.parent / "evaluation" / f"{preset}.jsonl"
                            )
                            .read_text()
                            .splitlines()
                        ]
                        old_agent = next(r for r in old if r["seed"] == seed)["agents"][
                            0
                        ]
                        assert summary["score"] == old_agent["score"]
                        assert summary["bombs"] == old_agent["bombs"]
                        assert summary["crates"] == old_agent["crates"]
                        summary["matches_original_score_bombs_crates"] = True
                        summaries.append(summary)
                        dump(
                            directory / f"{label}.json",
                            {
                                "summary": summary,
                                "trace": trace,
                                "explosions": explosions,
                            },
                        )
                        dump(directory / "summary.json", summaries)
                        print(json.dumps(summary), flush=True)
                    finally:
                        reset_framework_logging()
    assert hashes == {str(p): checksum(p) for p in protected}
    dump(directory / "integrity.json", {"unchanged": True, "games": len(summaries)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--analyze", action="store_true", help="read saved data only")
    args = parser.parse_args()
    if args.analyze:
        analyze(args.out)
    else:
        collect(args.out)


def analyze(directory: Path) -> None:
    """Audit immutable saved state and summarize games; never instantiate a learner."""
    from agent_code.dqn_agent.persistence import load_checkpoint, load_replay
    from agent_code.dqn_agent.rewards import Rewards
    from tests.test_training_dqn import assert_equal

    metrics = [json.loads(x) for x in (ROOT / "metrics.jsonl").read_text().splitlines()]
    chunks = [json.loads(x) for x in (ROOT / "chunks.jsonl").read_text().splitlines()]
    parent = Path("results/dqn/d6_20260917/pilot_lr3e-04_target1000/run_0")
    initial = ROOT / "checkpoints" / f"transition_{COUNTS[0]:09d}"
    a, b = (
        load_checkpoint(parent / "checkpoint.pt"),
        load_checkpoint(initial / "checkpoint.pt"),
    )
    for key in ("learner", "feature_rng", "transitions", "rounds_trained"):
        assert_equal(a[key], b[key])
    old_replay, _ = load_replay(parent / "replay.npz", a, a["schema_id"])
    first_replay, _ = load_replay(initial / "replay.npz", b, b["schema_id"])
    assert_equal(old_replay.snapshot(), first_replay.snapshot())
    net = QNetwork.load(parent / "q_net.npz", ENCODER)
    net2 = QNetwork.load(initial / "q_net.npz", ENCODER)
    for key in net.arrays:
        np.testing.assert_array_equal(net.arrays[key], net2.arrays[key])
    assert all(
        abs(row["epsilon"] - (0.3 - 0.25 * min(1, row["stage_position"] / 180000)))
        < 1e-12
        for row in metrics
    )
    assert all(row["transitions"] == row["stage_position"] + 50047 for row in metrics)
    assert all(row["updates"] == row["transitions"] // 4 - 1249 for row in metrics)
    audits: list[dict[str, Any]] = []
    for count in COUNTS:
        path = ROOT / "checkpoints" / f"transition_{count:09d}"
        state = load_checkpoint(path / "checkpoint.pt")
        replay, degraded = load_replay(path / "replay.npz", state, state["schema_id"])
        assert not degraded and state["exact_history"]
        rows = replay.snapshot()["rows"]
        done = rows["done"]
        assert not rows["phi_unit_next"][done].any()
        assert not rows["x_next"][done].any()
        assert not rows["mask_next"][done].any()
        assert rows["mask_next"][~done].any(axis=1).all()
        errors: list[float] = []
        for crate_aid, death_aid, c_coin in ((0, 0, 0.5), (0.2, -1, 0.3)):
            batch = replay.sample(
                2048,
                np.random.default_rng(77),
                gamma=0.99,
                c_coin=c_coin,
                crate_aid=crate_aid,
                death_aid=death_aid,
            )
            rewards = Rewards(0.99, c_coin, crate_aid, death_aid)
            expected = np.array(
                [
                    float(base)
                    + crate_aid * int(crates)
                    + death_aid * int(deaths)
                    + rewards.shaping(c_coin * float(phi), c_coin * float(nxt))
                    for base, crates, deaths, phi, nxt in zip(
                        batch.base,
                        batch.crates,
                        batch.deaths,
                        batch.phi_unit,
                        batch.phi_unit_next,
                        strict=True,
                    )
                ],
                dtype=np.float32,
            )
            np.testing.assert_allclose(batch.r, expected, atol=1e-6, rtol=0)
            errors.append(float(np.max(np.abs(batch.r - expected))))
        # Compare full retained training rounds to the original event/score records.
        complete_rounds = 0
        for record in metrics:
            subset = rows[rows["round_id"] == record["rounds_trained"]]
            if len(subset) != record["steps"]:
                continue
            assert int(subset["done"].sum()) == 1
            assert (
                float(subset["base"].sum()) == record["base_reward"] == record["coins"]
            )
            assert int(subset["crates"].sum()) == record["crates"]
            shaped = subset["base"].astype(float) + 0.5 * (
                0.99 * subset["phi_unit_next"].astype(float)
                - subset["phi_unit"].astype(float)
            )
            assert abs(float(shaped.sum()) - record["shaped_return"]) < 1e-5
            complete_rounds += 1
        audits.append(
            {
                "global_transitions": count,
                "stage_position": state["stage_position"],
                "updates": state["learner"]["updates"],
                "replay_length": len(replay),
                "terminals": int(done.sum()),
                "complete_rounds_matched": complete_rounds,
                "reward_max_errors": errors,
            }
        )
    evaluations: list[dict[str, Any]] = []
    for preset in ("crates-solo", "classic-solo", "coin-heaven-solo"):
        previous: list[int] | None = None
        for count in COUNTS:
            path = (
                ROOT
                / "checkpoints"
                / f"transition_{count:09d}"
                / "evaluation"
                / f"{preset}.jsonl"
            )
            rows = [json.loads(x) for x in path.read_text().splitlines()]
            coins = [r["agents"][0]["coins"] for r in rows]
            row = {
                "global_transitions": count,
                "stage_transitions": count - COUNTS[0],
                "preset": preset,
                "coins_by_seed": coins,
                "means": {
                    key: sum(r["agents"][0][key] for r in rows) / len(rows)
                    for key in ("coins", "crates", "bombs", "steps", "suicides")
                },
                "deaths": sum(not r["agents"][0]["survived"] for r in rows),
            }
            if previous is not None:
                row["paired_coin_change"] = [
                    y - x for x, y in zip(previous, coins, strict=True)
                ]
            evaluations.append(row)
            previous = coins
    games = json.loads((directory / "summary.json").read_text())
    examples: list[dict[str, Any]] = []
    for count, scenario, seed, step in (
        (150617, "loot-crate", 500, 25),
        (150617, "loot-crate", 500, 26),
        (350346, "loot-crate", 500, 120),
        (350346, "classic", 500, 63),
    ):
        source = directory / f"{count}_{scenario}_{seed}.json"
        data = json.loads(source.read_text())
        row = data["trace"][step - 1]
        examples.append(
            {
                "source": str(source),
                "global_transitions": count,
                "scenario": scenario,
                "seed": seed,
                **{
                    key: row[key]
                    for key in (
                        "step",
                        "pos",
                        "features",
                        "allowed",
                        "action",
                        "q_by_snapshot",
                        "score",
                        "score_after",
                        "coin_distance",
                        "reachable_coins",
                    )
                },
                "next_8_steps": [
                    {key: r[key] for key in ("step", "pos", "action", "score_after")}
                    for r in data["trace"][step - 1 : step + 7]
                ],
                "explosions_next_8_steps": [
                    r for r in data["explosions"] if step <= r["step"] < step + 8
                ],
            }
        )
    for game in games:
        path = (
            directory
            / f"{game['global_transitions']}_{game['scenario']}_{game['seed']}.json"
        )
        for row in json.loads(path.read_text())["trace"]:
            q = row["q_by_snapshot"][str(game["global_transitions"])]
            assert row["action"] in row["allowed"]
            assert q[row["action"]] == max(q.values())
    aggregate: list[dict[str, Any]] = []
    for count in COUNTS:
        for scenario in ("loot-crate", "classic"):
            chosen = [
                r
                for r in games
                if r["global_transitions"] == count and r["scenario"] == scenario
            ]
            keys = [
                k
                for k, v in chosen[0].items()
                if isinstance(v, int)
                and k not in ("seed", "global_transitions", "stage_transitions")
            ]
            aggregate.append(
                {
                    "global_transitions": count,
                    "scenario": scenario,
                    "games": len(chosen),
                    **{k: sum(r[k] for r in chosen) for k in keys},
                }
            )
    mixture = {
        name: {
            "rounds": sum(r["rounds"] for r in chunks if r["scenario"] == name),
            "transitions": sum(
                r["transitions"] for r in chunks if r["scenario"] == name
            ),
        }
        for name in ("loot-crate", "classic", "coin-heaven")
    }
    dump(
        directory / "audit.json",
        {
            "full_transfer_equal": True,
            "all_round_schedules_match": True,
            "epsilon_reference": {
                "0": 0.3,
                "150000": 0.3 - 0.25 * 150000 / 180000,
                "180000": 0.05,
                "300000": 0.05,
            },
            "mixture": mixture,
            "checkpoints": audits,
            "evaluations": evaluations,
            "bomb_behavior": aggregate,
            "same_observation_examples": examples,
            "all_diagnostic_actions_masked_greedy": True,
            "explored_training_actions": sum(r["explored_steps"] for r in metrics),
            "training_actions": sum(r["steps"] for r in metrics),
        },
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    assert all(
        checksum(Path(path)) == sha for path, sha in manifest["before_sha256"].items()
    )


if __name__ == "__main__":
    main()
