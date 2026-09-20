"""Unit tests for ``tabular_q_agent``'s transition bookkeeping, rewards and metrics.

The engine-driven counterparts are in ``test_tabular_q_agent_train.py``.
"""

import random
from pathlib import Path

import pytest

from agent_code.tabular_q_agent.config import (
    ENV_VAR,
    REPO_ROOT,
    Config,
    metrics_path,
    model_path,
)
from agent_code.tabular_q_agent.features import ENCODINGS
from agent_code.tabular_q_agent.learner import Learner, Selection
from agent_code.tabular_q_agent.metrics import append_record, read_records
from agent_code.tabular_q_agent.qtable import QTable
from agent_code.tabular_q_agent.rewards import Rewards
from agent_code.tabular_q_agent.transitions import BookkeepingError, Trainer

UP, RIGHT, DOWN, LEFT, WAIT, BOMB = range(6)
GAMMA = 0.9


def make_trainer(
    *,
    coin_potential: float = 0.0,
    crate_aid: float = 0.0,
    death_aid: float = 0.0,
) -> Trainer:
    learner = Learner(
        QTable.zeros(ENCODINGS["E1"]),
        gamma=GAMMA,
        alpha_omega=0.7,
        alpha_min=0.05,
        rng=random.Random(0),
    )
    rewards = Rewards(
        gamma=GAMMA,
        coin_potential=coin_potential,
        crate_aid=crate_aid,
        death_aid=death_aid,
    )
    trainer = Trainer(learner, rewards, epsilon=0.1, stage="unit")
    trainer.begin_round(1, ["rival"])
    return trainer


def act(
    trainer: Trainer,
    step: int,
    state: int,
    action: int,
    played: str,
    *,
    allowed: tuple[int, ...] = (UP, WAIT),
    coin_distance: int | None = None,
    round_number: int = 1,
) -> None:
    trainer.observe(round_number, step, state, allowed, coin_distance)
    trainer.chose(Selection(action, explored=False, unseen=False), played)


# --- transitions -----------------------------------------------------------


def test_a_survived_step_bootstraps_and_is_shaped() -> None:
    trainer = make_trainer(coin_potential=0.5)
    q = trainer.learner.table.q
    q[1, WAIT] = 2.0
    act(trainer, 1, 0, UP, "UP", coin_distance=3)  # phi = 0.5 / 4
    trainer.events_occurred(1, 1, "UP", ["MOVED_UP", "COIN_COLLECTED"])
    act(trainer, 2, 1, WAIT, "WAIT", coin_distance=1)  # phi' = 0.5 / 2

    target = 1.0 + (GAMMA * 0.25 - 0.125) + GAMMA * 2.0
    assert q[0, UP] == pytest.approx(target, rel=1e-6)  # first visit: alpha = 1


def test_a_terminal_step_has_no_bootstrap_and_zero_potential_after() -> None:
    trainer = make_trainer(coin_potential=0.5)
    trainer.learner.table.q[:] = 100.0
    act(trainer, 1, 0, WAIT, "WAIT", coin_distance=1)  # phi = 0.25
    trainer.finish("WAIT", ["WAITED", "SURVIVED_ROUND"])
    assert trainer.learner.table.q[0, WAIT] == pytest.approx(-0.25)


def test_a_survivors_last_step_is_counted_once() -> None:
    trainer = make_trainer()
    act(trainer, 1, 0, WAIT, "WAIT")
    trainer.events_occurred(1, 1, "WAIT", ["WAITED", "COIN_COLLECTED"])
    record = trainer.finish("WAIT", ["WAITED", "COIN_COLLECTED", "SURVIVED_ROUND"])

    assert record.coins == 1
    assert record.base_reward == 1.0
    assert record.event_counts == {
        "COIN_COLLECTED": 1,
        "SURVIVED_ROUND": 1,
        "WAITED": 1,
    }
    assert record.survived and not record.died
    assert trainer.learner.table.q[0, WAIT] == 1.0


def test_a_death_step_is_completed_by_end_of_round() -> None:
    trainer = make_trainer()
    act(trainer, 1, 0, WAIT, "WAIT")
    trainer.events_occurred(1, 1, "WAIT", ["WAITED"])
    act(trainer, 2, 1, BOMB, "BOMB")
    # No game_events_occurred for the death step; a posthumous kill follows.
    record = trainer.finish("BOMB", ["BOMB_DROPPED", "GOT_KILLED", "KILLED_OPPONENT"])

    assert record.died and not record.survived and not record.self_kill
    assert (record.steps, record.updates) == (2, 2)
    assert (record.kills, record.base_reward) == (1, 5.0)
    assert trainer.learner.table.q[1, BOMB] == 5.0
    assert trainer.pending is None


def test_a_suicide_pays_the_death_aid_once() -> None:
    trainer = make_trainer(death_aid=-1.0, crate_aid=0.25)
    act(trainer, 1, 0, WAIT, "WAIT")
    record = trainer.finish(
        "WAIT",
        ["WAITED", "BOMB_EXPLODED", "CRATE_DESTROYED", "KILLED_SELF", "GOT_KILLED"],
    )
    assert record.self_kill and record.died
    assert trainer.learner.table.q[0, WAIT] == pytest.approx(-1.0 + 0.25)
    assert record.base_reward == 0.0
    assert record.shaped_return == pytest.approx(-0.75)


def test_mutating_the_reported_list_later_changes_nothing() -> None:
    trainer = make_trainer()
    events = ["WAITED"]
    act(trainer, 1, 0, WAIT, "WAIT")
    trainer.events_occurred(1, 1, "WAIT", events)
    events.append("COIN_COLLECTED")
    act(trainer, 2, 1, WAIT, "WAIT")
    assert trainer.learner.table.q[0, WAIT] == 0.0


def test_the_round_record_describes_the_round() -> None:
    trainer = make_trainer()
    trainer.observe(1, 1, 0, (WAIT,), None)
    trainer.chose(Selection(WAIT, explored=False, unseen=True), "WAIT")
    trainer.events_occurred(1, 1, "WAIT", ["WAITED"])
    trainer.observe(1, 2, 1, (UP, WAIT), None)
    trainer.chose(Selection(UP, explored=True, unseen=True), "UP")
    trainer.events_occurred(1, 2, "UP", ["INVALID_ACTION"])
    record = trainer.finish("UP", ["INVALID_ACTION", "SURVIVED_ROUND"])

    assert (record.round, record.rounds_trained, record.stage) == (1, 1, "unit")
    assert record.opponents == ["rival"]
    assert (record.steps, record.updates, record.invalid) == (2, 2, 1)
    assert record.forced_fraction == 0.5
    assert (record.explored_steps, record.unseen_decisions) == (1, 1)
    assert (record.visited_states, record.new_states) == (2, 2)
    assert record.mean_alpha == 1.0
    assert record.epsilon == 0.1
    assert trainer.learner.table.meta == {"rounds_trained": 1, "steps_trained": 2}


def test_counters_continue_from_a_loaded_table() -> None:
    learner = Learner(
        QTable.zeros(ENCODINGS["E1"]),
        gamma=GAMMA,
        alpha_omega=0.7,
        alpha_min=0.05,
        rng=random.Random(0),
    )
    learner.table.meta = {"rounds_trained": 41, "steps_trained": 1000}
    trainer = Trainer(learner, Rewards(gamma=GAMMA), epsilon=0.0)
    trainer.begin_round(1, [])
    act(trainer, 1, 0, WAIT, "WAIT")
    record = trainer.finish("WAIT", ["WAITED", "SURVIVED_ROUND"])
    assert record.rounds_trained == 42
    assert learner.table.meta["steps_trained"] == 1001


# --- callbacks out of order --------------------------------------------------


def test_events_for_another_action_are_refused() -> None:
    trainer = make_trainer()
    act(trainer, 1, 0, WAIT, "WAIT")
    with pytest.raises(BookkeepingError):
        trainer.events_occurred(1, 1, "UP", ["MOVED_UP"])


def test_events_for_another_step_are_refused() -> None:
    trainer = make_trainer()
    act(trainer, 1, 0, WAIT, "WAIT")
    with pytest.raises(BookkeepingError):
        trainer.events_occurred(1, 2, "WAIT", ["WAITED"])


def test_events_reported_twice_are_refused() -> None:
    trainer = make_trainer()
    act(trainer, 1, 0, WAIT, "WAIT")
    trainer.events_occurred(1, 1, "WAIT", ["WAITED"])
    with pytest.raises(BookkeepingError):
        trainer.events_occurred(1, 1, "WAIT", ["WAITED"])


def test_acting_again_without_the_previous_steps_events_is_refused() -> None:
    trainer = make_trainer()
    act(trainer, 1, 0, WAIT, "WAIT")
    with pytest.raises(BookkeepingError):
        act(trainer, 2, 1, WAIT, "WAIT")


def test_end_of_round_that_rewrites_reported_events_is_refused() -> None:
    trainer = make_trainer()
    act(trainer, 1, 0, WAIT, "WAIT")
    trainer.events_occurred(1, 1, "WAIT", ["WAITED", "COIN_COLLECTED"])
    with pytest.raises(BookkeepingError):
        trainer.finish("WAIT", ["WAITED", "SURVIVED_ROUND"])


def test_callbacks_without_a_pending_action_are_refused() -> None:
    trainer = make_trainer()
    with pytest.raises(BookkeepingError):
        trainer.finish("WAIT", ["SURVIVED_ROUND"])
    with pytest.raises(BookkeepingError):
        trainer.events_occurred(1, 1, "WAIT", ["WAITED"])
    with pytest.raises(BookkeepingError):
        trainer.chose(Selection(WAIT, explored=False, unseen=False), "WAIT")


def test_a_new_round_with_a_pending_action_is_refused() -> None:
    trainer = make_trainer()
    act(trainer, 1, 0, WAIT, "WAIT")
    with pytest.raises(BookkeepingError):
        trainer.begin_round(2, [])


# --- rewards -----------------------------------------------------------------


def test_base_reward_is_the_score_delta_with_multiplicity() -> None:
    rewards = Rewards(gamma=GAMMA)
    events = ["COIN_COLLECTED", "KILLED_OPPONENT", "KILLED_OPPONENT", "CRATE_DESTROYED"]
    assert rewards.base(events) == 11.0
    assert rewards.base(["OPPONENT_ELIMINATED", "BOMB_DROPPED", "WAITED"]) == 0.0


def test_aids_count_crates_and_one_death() -> None:
    rewards = Rewards(gamma=GAMMA, crate_aid=0.1, death_aid=-2.0)
    assert rewards.aids(["CRATE_DESTROYED"] * 3) == pytest.approx(0.3)
    assert rewards.aids(["KILLED_SELF", "GOT_KILLED"]) == -2.0
    assert rewards.aids(["GOT_KILLED"]) == -2.0
    assert rewards.aids(["BOMB_DROPPED"]) == 0.0


def test_coin_potential_and_shaping() -> None:
    rewards = Rewards(gamma=GAMMA, coin_potential=0.6)
    assert rewards.potential(None) == 0.0
    assert rewards.potential(0) == 0.6
    assert rewards.potential(2) == pytest.approx(0.2)
    assert rewards.shaping(0.2, 0.3) == pytest.approx(GAMMA * 0.3 - 0.2)


# --- metrics and config ------------------------------------------------------


def test_records_round_trip_through_jsonl(tmp_path: Path) -> None:
    trainer = make_trainer()
    act(trainer, 1, 0, WAIT, "WAIT")
    first = trainer.finish("WAIT", ["WAITED", "SURVIVED_ROUND"])
    trainer.begin_round(2, [])
    act(trainer, 1, 1, WAIT, "WAIT", round_number=2)
    second = trainer.finish("WAIT", ["WAITED", "GOT_KILLED"])

    path = tmp_path / "logs" / "metrics.jsonl"
    append_record(path, first)
    append_record(path, second)
    assert read_records(path) == [first, second]


def test_training_parameters_default_and_validate() -> None:
    config = Config()
    assert (config.epsilon, config.coin_potential, config.save_every) == (0.1, 0.5, 1)
    assert (config.crate_aid, config.death_aid, config.stage) == (0.0, 0.0, "")
    override = '{"epsilon": 0, "death_aid": -1, "stage": "coins", "save_every": 5}'
    assert Config.from_env({ENV_VAR: override}).save_every == 5
    for bad in (
        '{"epsilon": 1.5}',
        '{"coin_potential": -1}',
        '{"death_aid": "x"}',
        '{"save_every": 0}',
        '{"save_every": 2.5}',
        '{"stage": 3}',
    ):
        with pytest.raises(ValueError):
            Config.from_env({ENV_VAR: bad})


def test_relative_paths_are_taken_from_the_repository_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    environ = {
        "TABULAR_Q_AGENT_MODEL": "results/run/q.npz",
        "TABULAR_Q_AGENT_METRICS": "results/run/metrics.jsonl",
    }
    assert model_path(environ) == (REPO_ROOT / "results/run/q.npz", True)
    assert metrics_path(environ) == REPO_ROOT / "results/run/metrics.jsonl"
    assert (REPO_ROOT / "main.py").exists()

    absolute_model = tmp_path / "q.npz"
    absolute_metrics = tmp_path / "metrics.jsonl"
    assert absolute_model.is_absolute() and absolute_metrics.is_absolute()
    environ = {
        "TABULAR_Q_AGENT_MODEL": str(absolute_model),
        "TABULAR_Q_AGENT_METRICS": str(absolute_metrics),
    }
    assert model_path(environ) == (absolute_model, True)
    assert metrics_path(environ) == absolute_metrics
