"""Training accounting and exact round-boundary resume against the real engine."""

import copy
import json
import random
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

import settings
from agent_code.dqn_agent.config import ENV_VAR, METRICS_ENV_VAR, MODEL_ENV_VAR
from agent_code.dqn_agent.core.world_model import ACTIONS
from agent_code.dqn_agent.metrics import read_records
from agent_code.dqn_agent.network import QNetwork
from items import Bomb
from tournament.engine import quiet_logging, reset_framework_logging
from training.world import TrainingWorld, create_training_world, play_round

if TYPE_CHECKING:
    from agent_code.dqn_agent.train import Trainer

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for variable in (ENV_VAR, MODEL_ENV_VAR, METRICS_ENV_VAR):
        monkeypatch.delenv(variable, raising=False)
    reset_framework_logging()
    with quiet_logging():
        yield
    reset_framework_logging()


def configure(monkeypatch: pytest.MonkeyPatch, path: Path, **params: object) -> None:
    monkeypatch.setenv(MODEL_ENV_VAR, str(path / "q_net.npz"))
    monkeypatch.setenv(METRICS_ENV_VAR, str(path / "metrics.jsonl"))
    monkeypatch.setenv(
        ENV_VAR,
        json.dumps(
            {
                "seed": 0,
                "init_seed": 0,
                "replay_size": 3000,
                "replay_save_every": 1,
                **params,
            }
        ),
    )


def trainer(world: TrainingWorld) -> "Trainer":
    return cast("Trainer", cast(Any, world).agents[0].backend.runner.fake_self.trainer)


def script(
    monkeypatch: pytest.MonkeyPatch, subject: "Trainer", actions: Sequence[str] = ()
) -> None:
    assert all(a in ("WAIT", "BOMB") for a in actions)
    planned = iter(actions)

    def choose(x: NDArray[np.float32], allowed: Sequence[int]) -> tuple[int, bool]:
        return ACTIONS.index(next(planned, "WAIT")), False

    monkeypatch.setattr(subject, "choose", choose)


def test_survivor_and_warmup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure(monkeypatch, tmp_path)
    world = create_training_world(("dqn_agent", "peaceful_agent"), "empty", 0)
    subject = trainer(world)
    before = subject.learner.export().arrays
    script(monkeypatch, subject)
    play_round(world)
    record = subject.last_record
    rows = subject.replay.snapshot()["rows"]
    assert subject.transitions == record["steps"] == 400
    assert rows["done"].sum() == 1 and rows[-1]["done"]
    assert record["event_counts"] == {"WAITED": 400, "SURVIVED_ROUND": 1}
    assert subject.bookkeeper.pending is None
    assert subject.learner.updates == record["updates"] == 0
    for name, value in before.items():
        np.testing.assert_array_equal(value, subject.learner.export().arrays[name])
    assert (
        QNetwork.load(tmp_path / "q_net.npz", subject.agent.encoder).meta["transitions"]
        == 400
    )
    assert read_records(tmp_path / "metrics.jsonl")[0]["mean_loss"] is None


def test_suicide(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure(monkeypatch, tmp_path, death_aid=-1)
    world = create_training_world(("dqn_agent",), "empty", 0)
    subject = trainer(world)
    script(monkeypatch, subject, ["BOMB"])
    play_round(world)
    rows = subject.replay.snapshot()["rows"]
    assert subject.transitions == 5
    assert rows["done"].tolist() == [False] * 4 + [True]
    assert rows[-1]["deaths"] == 1
    batch = subject.replay.sample(
        100, np.random.default_rng(0), gamma=0.99, death_aid=-1
    )
    assert np.all(batch.r[batch.done] == -1)
    assert subject.last_record["shaped_return"] == -1
    assert subject.last_record["event_counts"]["GOT_KILLED"] == 1
    assert subject.last_record["event_counts"]["KILLED_SELF"] == 1


def test_bomb_aid_pays_for_booked_crates_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The drop step earns ``bomb_aid`` per live crate in its blast; the four
    waiting steps that follow earn nothing from it."""
    configure(monkeypatch, tmp_path, bomb_aid=0.25, death_aid=-1)
    world: Any = create_training_world(("dqn_agent",), "empty", 0)
    subject = trainer(world)
    script(monkeypatch, subject, ["BOMB"])
    world.new_round()
    learner = world.agents[0]
    learner.x, learner.y = 1, 1
    for crate in ((2, 1), (3, 1), (1, 2)):
        world.arena[crate] = 1
    while world.running:
        world.do_step()
    rows = subject.replay.snapshot()["rows"]
    assert rows["bombs"].tolist() == [3, 0, 0, 0, 0]
    batch = subject.replay.sample(
        200, np.random.default_rng(0), gamma=0.99, bomb_aid=0.25
    )
    first = batch.r[batch.transition_id == 0]
    assert len(first) and np.allclose(first, 0.75)
    assert np.allclose(batch.r[batch.transition_id > 0], 0.0)


def test_attack_aid_pays_only_for_a_bomb_with_an_opponent_in_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The kill anchor is the crate anchor's twin: paid on the drop, scaled by
    what the bomb does to an opponent (1 pressure, 2 trap), and zero for a bomb
    dropped with nobody in reach."""
    configure(monkeypatch, tmp_path, attack_aid=0.5, bomb_aid=0.0)
    world: Any = create_training_world(("dqn_agent", "peaceful_agent"), "empty", 0)
    subject = trainer(world)
    script(monkeypatch, subject, ["BOMB"])
    world.new_round()
    learner, other = world.agents
    # A dead end: the opponent stands in the blast with a wall behind it.
    learner.x, learner.y = 1, 1
    other.x, other.y = 3, 1
    for crate in ((1, 2), (3, 2), (2, 2)):
        world.arena[crate] = 1
    while world.running:
        world.do_step()
    rows = subject.replay.snapshot()["rows"]
    attacks = rows["attacks"].tolist()
    assert attacks[0] > 0, "the drop step saw an opponent in the blast"
    assert attacks[1:] == [0] * (len(attacks) - 1), "only the drop is paid"
    batch = subject.replay.sample(
        200, np.random.default_rng(0), gamma=0.99, attack_aid=0.5
    )
    drop = batch.r[batch.transition_id == 0] - batch.base[batch.transition_id == 0]
    assert len(drop) and np.allclose(drop, 0.5 * attacks[0])
    # The same buffer with the aid switched off pays nothing for it.
    off = subject.replay.sample(200, np.random.default_rng(0), gamma=0.99)
    np.testing.assert_allclose(off.r, off.base, atol=1e-6)


def test_spot_potential_is_a_potential_of_the_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Stored as a unit potential, so a changed coefficient stays consistent
    with ``Rewards`` over the whole buffer, and a round's shaping telescopes."""
    from agent_code.dqn_agent.rewards import Rewards

    configure(monkeypatch, tmp_path, spot_potential=0.2, c_coin=0.0)
    world = create_training_world(("dqn_agent",), "loot-crate", 3)
    subject = trainer(world)
    play_round(world)
    rows = subject.replay.snapshot()["rows"]
    assert (rows["spot_unit"] > 0).any(), "the board has bombing spots"
    for coefficient in (0.0, 0.2, 1.0):
        rewards = Rewards(0.99, spot_potential=coefficient)
        batch = subject.replay.sample(
            256,
            np.random.default_rng(1),
            gamma=0.99,
            spot_potential=coefficient,
        )
        expected = [
            rewards.base([]) + rewards.shaping(coefficient * phi, coefficient * nxt)
            for phi, nxt in zip(batch.spot_unit, batch.spot_unit_next, strict=True)
        ]
        np.testing.assert_allclose(batch.r - batch.base, expected, atol=2e-6, rtol=1e-6)
    assert rows["spot_unit_next"][rows["done"]].tolist() == [0.0]


def test_hunt_potential_is_paid_on_closing_the_distance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The term that gives the agent something to do once the board is empty.

    It is a potential of the state, so a step that halves the distance to the
    opponent earns ``gamma*c/(1+d') - c/(1+d)`` and a round that ends where it
    began earns nothing at all; and it is recomputed at sample time, so turning
    it off leaves the rest of the reward untouched.
    """
    from agent_code.dqn_agent.rewards import Rewards

    configure(monkeypatch, tmp_path, hunt_potential=0.3, c_coin=0.0)
    # ``empty`` has no crates and no coins, so the board is stripped from the
    # first step and the potential is live throughout.
    world = create_training_world(("dqn_agent", "peaceful_agent"), "empty", 1)
    subject = trainer(world)
    play_round(world)
    rows = subject.replay.snapshot()["rows"]
    assert (rows["hunt_unit"] > 0).any(), "an opponent was reachable"
    assert rows["hunt_unit_next"][rows["done"]].tolist() == [0.0]
    for coefficient in (0.0, 0.3, 1.0):
        rewards = Rewards(0.99, hunt_potential=coefficient)
        batch = subject.replay.sample(
            256, np.random.default_rng(2), gamma=0.99, hunt_potential=coefficient
        )
        expected = [
            rewards.shaping(coefficient * phi, coefficient * nxt)
            for phi, nxt in zip(batch.hunt_unit, batch.hunt_unit_next, strict=True)
        ]
        np.testing.assert_allclose(batch.r - batch.base, expected, atol=2e-6)


def test_posthumous_kill(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure(monkeypatch, tmp_path)
    world: Any = create_training_world(("dqn_agent", "peaceful_agent"), "empty", 0)
    subject = trainer(world)
    script(monkeypatch, subject, ["BOMB"])
    world.new_round()
    learner, walker = world.agents
    for crate in ((2, 1), (4, 1), (3, 2)):
        world.arena[crate] = 1
    learner.x, learner.y = 1, 1
    walker.x, walker.y = 3, 1
    world.bombs.append(Bomb((1, 3), walker, 1, settings.BOMB_POWER, walker.bomb_sprite))
    while world.running:
        world.do_step()
    assert learner.dead and walker.dead and world.step == 5
    rows = subject.replay.snapshot()["rows"]
    assert (
        len(rows) == 2 and rows[-1]["done"] and rows[-1]["base"] == learner.score == 5
    )
    assert subject.last_record["base_reward"] == 5


def test_eight_rounds_accounting_and_canonical_masks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configure(monkeypatch, tmp_path, epsilon_start=0.3, epsilon_end=0.3)
    for seed in range(8):
        world: Any = create_training_world(
            ("dqn_agent",) + ("rule_based_agent",) * 3, "classic", seed
        )
        subject = trainer(world)
        before = subject.transitions
        play_round(world)
        record = subject.last_record
        rows = subject.replay.snapshot()["rows"]
        rows = rows[rows["transition_id"] >= before]
        assert rows["base"].sum() == record["base_reward"] == world.agents[0].score
        assert len(rows) == record["steps"] == world.agents[0].statistics["steps"]
        assert rows["done"].sum() == 1
        assert subject.bookkeeper.pending is None
        assert record["rounds_trained"] == seed + 1
        assert np.all(rows["x"][np.arange(len(rows)), rows["a"]] == 1)
        np.testing.assert_array_equal(
            rows["x_next"][:, :6].astype(bool), rows["mask_next"]
        )
        reset_framework_logging()


def assert_state_equal(left: object, right: object) -> None:
    if isinstance(left, dict):
        a, b = cast(dict[str, Any], left), cast(dict[str, Any], right)
        assert a.keys() == b.keys()
        for key in a:
            assert_state_equal(a[key], b[key])
    elif isinstance(left, (list, tuple)):
        a_seq, b_seq = cast(list[Any], left), cast(list[Any], right)
        assert len(a_seq) == len(b_seq)
        for a, b in zip(a_seq, b_seq, strict=True):
            assert_state_equal(a, b)
    elif torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(cast(NDArray[np.generic], left), right)
    else:
        assert left == right


def test_split_resume_exact_with_controlled_world_and_opponent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from agent_code.peaceful_agent import callbacks as opponent

    def setup(agent: object) -> None:
        pass

    def act(agent: object, state: dict[str, Any]) -> str:
        # Explicit deterministic opponent: no entropy seeding, no hidden state.
        return random.Random(9000 + state["round"] * 400 + state["step"]).choice(
            ACTIONS[:4]
        )

    monkeypatch.setattr(opponent, "setup", setup)
    monkeypatch.setattr(opponent, "act", act)
    outcomes: list[Trainer] = []
    for split in (False, True):
        directory = tmp_path / str(split)
        configure(monkeypatch, directory, warmup=8, target_every=37, replay_size=600)
        world: Any = create_training_world(
            ("dqn_agent", "peaceful_agent"), "coin-heaven", 70
        )
        for round_index in range(5):
            play_round(world)
            if split and round_index == 2:
                subject = trainer(world)
                subject.save(full=True)
                state = copy.deepcopy(subject.learner.training_state())
                replay = subject.replay.snapshot()
                feature_rng = subject.agent.rng.getstate()
                count = subject.transitions
                world_rng = copy.deepcopy(world.rng.bit_generator.state)
                reset_framework_logging()
                world = create_training_world(
                    ("dqn_agent", "peaceful_agent"), "coin-heaven", 70
                )
                world.rng.bit_generator.state = world_rng
                world.round = 3
                restored = trainer(world)
                assert restored.resume_mode == "exact-resume"
                assert_state_equal(state, restored.learner.training_state())
                assert_state_equal(replay, restored.replay.snapshot())
                assert feature_rng == restored.agent.rng.getstate()
                assert restored.transitions == restored.stage_position == count
        outcomes.append(trainer(world))
        reset_framework_logging()
    first, second = outcomes
    assert (
        first.transitions
        == second.transitions
        == first.stage_position
        == second.stage_position
    )
    assert first.rounds_trained == second.rounds_trained == 5
    assert first.learner.updates > 0
    assert_state_equal(first.learner.training_state(), second.learner.training_state())
    assert_state_equal(first.replay.snapshot(), second.replay.snapshot())
    assert first.agent.rng.getstate() == second.agent.rng.getstate()
    assert first.last_record["epsilon"] == second.last_record["epsilon"]


def test_resume_accepts_a_checkpoint_written_before_a_field_existed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Adding a coefficient with a default must not orphan running training.

    A checkpoint that predates the field was trained with the default value, so
    filling it in is the truth, not a guess. A checkpoint that *disagrees* about
    a field it does name is still refused.
    """
    from agent_code.dqn_agent.persistence import load_checkpoint, save_checkpoint

    configure(monkeypatch, tmp_path)
    world = create_training_world(("dqn_agent",), "empty", 0)
    subject = trainer(world)
    script(monkeypatch, subject, ["WAIT"])
    play_round(world)
    subject.save(full=True)

    path = tmp_path / "checkpoint.pt"
    state = load_checkpoint(path)
    written = dict(state["config"])
    assert written.pop("attack_aid") == 0.0
    save_checkpoint(path, {**state, "config": written})
    resumed = create_training_world(("dqn_agent",), "empty", 0)
    restored = trainer(resumed)
    assert restored.transitions == subject.transitions
    assert restored.resume_mode == "exact-resume"

    state = load_checkpoint(path)
    save_checkpoint(path, {**state, "config": {**state["config"], "gamma": 0.5}})
    with pytest.raises(ValueError, match="mismatch"):
        create_training_world(("dqn_agent",), "empty", 0)


def test_hunt_potential_is_zero_while_anything_is_collectable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The gate: chasing opponents must never compete with collecting.

    On a board that still has crates and coins the term is identically zero, so
    the early game is exactly the game it was without it; it switches on only
    when there is nothing else to do.
    """
    configure(monkeypatch, tmp_path, hunt_potential=0.3)
    world = create_training_world(("dqn_agent", "peaceful_agent"), "loot-crate", 5)
    subject = trainer(world)
    play_round(world)
    rows = subject.replay.snapshot()["rows"]
    live = rows[:40]  # the opening, with crates and coins still on the board
    assert not live["hunt_unit"].any(), "the potential was paid too early"
    assert not live["hunt_unit_next"].any()
    batch = subject.replay.sample(
        128, np.random.default_rng(4), gamma=0.9, hunt_potential=0.3
    )
    assert np.isfinite(batch.r).all()
