"""Engine-driven tests of ``tabular_q_agent``'s training callbacks.

These run the real engine, so they check the bookkeeping against what the
framework actually delivers rather than against this code's reading of it:
a survivor's doubled final step, a death step completed by ``end_of_round``,
a kill scored after death, and exact score accounting over real games.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import settings
from agent_code.tabular_q_agent.metrics import read_records
from agent_code.tabular_q_agent.qtable import QTable
from items import Bomb
from tests.tabular_q_worlds import agent_self, script, spy_updates, training_env
from tournament.engine import quiet_logging, reset_framework_logging
from training.world import create_training_world, play_round


@pytest.fixture(autouse=True)
def _clean_logging() -> Iterator[None]:
    reset_framework_logging()
    with quiet_logging():
        yield
    reset_framework_logging()


def test_a_survivor_gets_one_update_per_step_and_one_terminal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model, metrics = training_env(monkeypatch, tmp_path)
    world: Any = create_training_world(
        ("tabular_q_agent", "peaceful_agent"), "empty", 0
    )
    agent = agent_self(world)
    script(monkeypatch, agent, [])  # WAIT for the whole round
    updates = spy_updates(monkeypatch, agent)
    play_round(world)

    (record,) = read_records(metrics)
    assert world.step == settings.MAX_STEPS
    assert record.steps == record.updates == len(updates) == settings.MAX_STEPS
    assert [call.next_state for call in updates].count(None) == 1
    assert updates[-1].next_state is None
    assert record.survived and not record.died
    # The final step's WAITED arrives in both callbacks; it counts once.
    assert record.event_counts == {"SURVIVED_ROUND": 1, "WAITED": settings.MAX_STEPS}
    assert agent.trainer is not None and agent.trainer.pending is None
    assert QTable.load(model, agent.encoding).meta["rounds_trained"] == 1


def test_a_suicide_is_one_terminal_transition_with_one_death_aid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, metrics = training_env(monkeypatch, tmp_path, death_aid=-1.0)
    world: Any = create_training_world(("tabular_q_agent",), "empty", 0)
    agent = agent_self(world)
    script(monkeypatch, agent, ["BOMB"])  # then wait on it
    updates = spy_updates(monkeypatch, agent)
    play_round(world)

    (record,) = read_records(metrics)
    death_step = settings.BOMB_TIMER + 1
    assert world.step == death_step
    assert record.died and record.self_kill and not record.survived
    assert record.steps == record.updates == len(updates) == death_step
    assert [call.next_state is None for call in updates] == [False] * 4 + [True]
    # KILLED_SELF and GOT_KILLED both arrive, and only in end_of_round.
    assert updates[-1].reward == -1.0
    assert record.event_counts["KILLED_SELF"] == record.event_counts["GOT_KILLED"] == 1
    assert record.event_counts["WAITED"] == death_step - 1
    assert (record.base_reward, record.shaped_return) == (0.0, -1.0)


def test_a_kill_after_death_is_credited_to_the_death_transition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The learner bombs at (1, 1) next to a walker boxed in by crates at
    (3, 1). The walker's own bomb at (1, 3) kills the learner after step 2;
    the learner's bomb goes off after step 5, blasts through the crate at
    (2, 1) and kills the walker."""
    _, metrics = training_env(monkeypatch, tmp_path)
    world: Any = create_training_world(
        ("tabular_q_agent", "peaceful_agent"), "empty", 0
    )
    agent = agent_self(world)
    script(monkeypatch, agent, ["BOMB"])
    updates = spy_updates(monkeypatch, agent)

    world.new_round()
    learner, walker = world.agents
    for crate in ((2, 1), (4, 1), (3, 2)):
        world.arena[crate] = 1
    learner.x, learner.y = 1, 1
    walker.x, walker.y = 3, 1
    world.bombs.append(Bomb((1, 3), walker, 1, settings.BOMB_POWER, walker.bomb_sprite))
    while world.running:
        world.do_step()

    (record,) = read_records(metrics)
    assert learner.dead and walker.dead
    assert world.step == 5  # the stock training rule would have stopped at 2
    assert learner.score == 5
    assert record.died and not record.self_kill
    assert (record.steps, record.updates) == (2, 2)
    assert (record.kills, record.crates, record.base_reward) == (1, 2, 5.0)
    assert updates[-1].next_state is None
    assert updates[-1].reward == 5.0


def test_rewards_add_up_to_the_engine_score_in_real_games(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Rounds against three rule_based agents with exploration on.

    Each round builds a new world, which reloads the table the previous round
    saved -- so the counters must continue across worlds too.
    """
    rounds = 8
    _, metrics = training_env(monkeypatch, tmp_path, epsilon=0.3)
    lineup = ("tabular_q_agent",) + ("rule_based_agent",) * 3
    for seed in range(rounds):
        world: Any = create_training_world(lineup, "classic", seed)
        agent = agent_self(world)
        play_round(world)
        record = read_records(metrics)[-1]
        learner = world.agents[0]

        assert record.base_reward == learner.score, f"seed {seed}"
        assert record.steps == record.updates == learner.statistics["steps"]
        assert record.coins == learner.statistics["coins"]
        assert record.kills == learner.statistics["kills"]
        assert record.died == learner.dead
        assert agent.trainer is not None and agent.trainer.pending is None
        reset_framework_logging()

    records = read_records(metrics)
    assert [r.rounds_trained for r in records] == list(range(1, rounds + 1))
