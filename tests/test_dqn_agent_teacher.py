"""The training-only teacher policy: identical to bfs_agent, absent at play.

Two things must hold for the teacher to be honest. It has to be the *same*
policy as the benchmark agent, or a run that reports "20% of steps played by
the search agent" is reporting something else; and it must not exist on the
inference path, where the submitted agent may only read its own network.
"""

import logging
import random
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent_code.bfs_agent import callbacks as bfs
from agent_code.dqn_agent.core.world_model import ACTIONS, Observation
from agent_code.dqn_agent.teacher import Teacher
from tournament.engine import (
    WorldConfig,
    create_world,
    quiet_logging,
    reset_framework_logging,
)


def engine_states(scenario: str, seed: int, lineup: tuple[str, ...]) -> list[Any]:
    states: list[Any] = []
    with quiet_logging():
        world: Any = create_world(WorldConfig(lineup, scenario, seed))
        try:
            world.new_round()
            world.user_input = None
            while world.running:
                for agent in world.active_agents:
                    states.append(world.get_state_for_agent(agent))
                world.do_step()
        finally:
            reset_framework_logging()
    return states


@pytest.mark.parametrize(
    "scenario,seed,lineup",
    [
        ("classic", 11, ("bfs_agent", "rule_based_agent", "rule_based_agent")),
        ("loot-crate", 12, ("bfs_agent",)),
        ("coin-heaven", 13, ("bfs_agent",)),
    ],
)
def test_teacher_reproduces_bfs_agent_on_engine_states(
    scenario: str, seed: int, lineup: tuple[str, ...]
) -> None:
    """Same states, same private RNG seed, same actions -- the teacher is the
    benchmark agent's decision rule, not an approximation of it."""
    states = engine_states(scenario, seed, lineup)
    assert len(states) >= 50
    teacher = Teacher(seed=7)
    reference = cast(Any, SimpleNamespace(logger=logging.getLogger("test.bfs")))
    # bfs_agent seeds its own generator from Params; give both the same stream.
    bfs.setup(reference)
    reference.rng = random.Random(7)
    agreed = 0
    for state in states:
        mine = teacher.act(Observation.from_game_state(state))
        theirs = bfs.act(reference, state)
        assert mine in ACTIONS
        agreed += mine == theirs
    assert agreed == len(states), f"{len(states) - agreed} decisions differ"


def test_round_state_is_reset_between_rounds() -> None:
    """The coin route is a within-round commitment; carrying it into the next
    round would make the teacher follow a plan for a board that is gone."""
    states = engine_states("coin-heaven", 14, ("bfs_agent",))
    teacher = Teacher(seed=1)
    first = Observation.from_game_state(states[10])
    teacher.act(first)
    route = teacher.route
    later = first
    teacher.round = later.round - 1
    teacher.act(later)
    assert teacher.round == later.round
    assert teacher.route is not route, "a new round starts a new coin route"


def test_inference_never_imports_the_teacher() -> None:
    """The play path loads the network and nothing else: no search, no torch."""
    code = (
        "import sys;"
        "import agent_code.dqn_agent.callbacks as c;"
        "assert 'agent_code.dqn_agent.teacher' not in sys.modules, 'teacher';"
        "assert 'torch' not in sys.modules, 'torch';"
        "print('clean')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout
