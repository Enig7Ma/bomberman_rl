"""D4 accounting and exact round-boundary resume against the real engine."""

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
