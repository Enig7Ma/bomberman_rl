"""Tests for ``training.world``'s stop rule, driven through the real engine."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import settings
from items import Bomb
from tests.tabular_q_worlds import agent_self, script, training_env
from tournament.engine import quiet_logging, reset_framework_logging
from training.world import create_training_world, play_round

LEARNER_AND_WALKER = ("tabular_q_agent", "peaceful_agent")


@pytest.fixture(autouse=True)
def _clean_logging() -> Iterator[None]:
    reset_framework_logging()
    with quiet_logging():
        yield
    reset_framework_logging()


def test_only_the_first_seats_train(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    training_env(monkeypatch, tmp_path)
    world: Any = create_training_world(LEARNER_AND_WALKER, "empty", seed=0)
    assert [agent.train for agent in world.agents] == [True, False]


def test_a_dead_learners_bomb_keeps_the_round_going(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The learner bombs and waits on its bomb: it dies after step 5. Its
    explosion is still dangerous in step 6 and turns to smoke in step 7, so
    the round ends after step 7 -- the stock rule would end it after step 5.
    The walker starts in another corner, far outside the blast."""
    training_env(monkeypatch, tmp_path)
    world: Any = create_training_world(LEARNER_AND_WALKER, "empty", seed=0)
    script(monkeypatch, agent_self(world), ["BOMB"])
    play_round(world)

    learner, walker = world.agents
    assert learner.dead and not walker.dead
    assert world.step == settings.BOMB_TIMER + 1 + settings.EXPLOSION_TIMER


def test_a_learner_without_bombs_out_ends_the_round_when_it_dies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    training_env(monkeypatch, tmp_path)
    world: Any = create_training_world(LEARNER_AND_WALKER, "empty", seed=0)
    script(monkeypatch, agent_self(world), [])
    world.new_round()
    learner, walker = world.agents
    # The walker's bomb under the learner explodes after the first step.
    world.bombs.append(
        Bomb((learner.x, learner.y), walker, 0, settings.BOMB_POWER, walker.bomb_sprite)
    )
    while world.running:
        world.do_step()

    assert learner.dead and not walker.dead
    assert world.step == 1
