"""Differential tests: ``bfs_agent``'s danger timeline against the real engine.

Everything the agent's safety layer (and later the learned agents' safety
layer) assumes about bomb timing is checked here against ``BombeRLeWorld``
itself, not against the rules PDF and not against a re-implementation. Each
case places bombs/crates/agents in a real world, forecasts from the state the
engine would hand an agent, then ticks the engine and compares offset by offset:

- which cells a dangerous explosion covers after each action (the engine's own
  kill criterion in ``evaluate_explosions``),
- whether a stationary agent actually dies, and at which offset,
- which tiles a bomb blocks, and when blasted crates become enterable.

A tick is ``do_step`` with every agent waiting: ``poll_and_run_agents`` is the
only part skipped, and a waiting agent's action is a no-op in the engine.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from agent_code.bfs_agent.world_model import (
    BOMB_TIMER,
    EXPLOSION_TIMER,
    Geometry,
    Observation,
    Pos,
    Timeline,
    crate_cells,
    danger_timeline,
    own_bomb,
)
from environment import BombeRLeWorld
from items import Bomb
from tests.bfs_boards import arena
from tournament.engine import WorldConfig, create_world, reset_framework_logging
from tournament.framework import AgentView, agents_of

# Enough ticks for a timer-4 bomb to explode, burn out and turn to smoke.
TICKS = BOMB_TIMER + EXPLOSION_TIMER + 3


class ExplosionView(Protocol):
    blast_coords: list[Pos]

    def is_dangerous(self) -> bool: ...


class BombView(Protocol):
    x: int
    y: int

    def get_blast_coords(self, arena: NDArray[np.int64]) -> list[Pos]: ...


@dataclass(frozen=True)
class Case:
    name: str
    bombs: Sequence[tuple[Pos, int]]
    crates: Sequence[Pos] = ()
    # Stationary agents, placed on these cells; at most four.
    sitters: Sequence[Pos] = ((1, 1),)


CASES = [
    *[
        Case(f"timer {k}", bombs=[((5, 5), k)], sitters=[(5, 5), (5, 8), (5, 9)])
        for k in range(BOMB_TIMER + 1)
    ],
    # Odd rows/columns hold stone walls, so a bomb at (4, 5) sits between
    # pillars: its vertical blast is blocked immediately.
    # (4, 7) is two cells away but behind the pillar at (4, 6).
    Case("walls stop blasts", bombs=[((4, 5), 1)], sitters=[(4, 5), (4, 7), (7, 5)]),
    Case(
        "blast passes through crates",
        bombs=[((5, 1), 2)],
        crates=[(6, 1), (7, 1), (5, 2)],
        sitters=[(8, 1), (5, 4), (9, 1)],
    ),
    Case(
        "overlapping blasts",
        bombs=[((5, 5), 1), ((7, 5), 3)],
        sitters=[(6, 5), (5, 5), (9, 5), (3, 5)],
    ),
    # (5, 7) is inside the first bomb's blast. A chain reaction would detonate
    # it early; the engine does not, so it must still explode on its own timer.
    Case("no chain detonation", bombs=[((5, 5), 0), ((5, 7), 4)], sitters=[(5, 9)]),
    Case("board edge", bombs=[((1, 1), 2)], sitters=[(1, 3), (4, 1), (1, 5)]),
]


@pytest.fixture(autouse=True)
def _clean_logging() -> Iterator[None]:
    reset_framework_logging()
    yield
    reset_framework_logging()


def build_world(case: Case) -> tuple[BombeRLeWorld, list[AgentView]]:
    lineup = ("peaceful_agent",) * len(case.sitters)
    world = create_world(WorldConfig(lineup=lineup, scenario="empty", seed=0))
    world.new_round()
    # Normally set by ``do_step``; ``get_state_for_agent`` reads it.
    world.user_input = "WAIT"
    world.arena = arena(case.crates)
    world.coins = []
    agents = agents_of(world)
    for agent, pos in zip(agents, case.sitters, strict=True):
        agent.x, agent.y = pos
    owner = world.agents[0]
    for pos, timer in case.bombs:
        world.bombs.append(Bomb(pos, owner, timer, 3, owner.bomb_sprite))
    return world, agents


def tick(world: BombeRLeWorld) -> None:
    """``do_step`` with every agent waiting."""
    world.collect_coins()
    world.update_explosions()
    world.update_bombs()
    world.evaluate_explosions()


def engine_lethal(world: BombeRLeWorld) -> set[Pos]:
    cells: set[Pos] = set()
    for explosion in cast(list[ExplosionView], world.explosions):
        if explosion.is_dangerous():
            cells.update((int(x), int(y)) for x, y in explosion.blast_coords)
    return cells


def engine_bomb_tiles(world: BombeRLeWorld) -> set[Pos]:
    return {(int(b.x), int(b.y)) for b in cast(list[BombView], world.bombs)}


def observe(world: BombeRLeWorld, seat: int) -> Observation:
    """The observation the engine would hand the agent in ``seat``."""
    state = cast(dict[str, Any] | None, world.get_state_for_agent(world.agents[seat]))
    assert state is not None, "the observing agent must be alive"
    return Observation.from_game_state(state)


def forecast(world: BombeRLeWorld, seat: int = 0) -> Timeline:
    obs = observe(world, seat)
    geometry = Geometry(obs.field)
    return danger_timeline(
        geometry, crate_cells(obs.field), obs.bombs, obs.explosion_map
    )


def compare(world: BombeRLeWorld, agents: list[AgentView], timeline: Timeline) -> None:
    """Tick the engine and compare against ``timeline`` at every offset."""
    first_lethal = {
        (agent.x, agent.y): next(
            iter(timeline.lethal_offsets((agent.x, agent.y))), None
        )
        for agent in agents
    }
    died_at: dict[Pos, int] = {}
    for offset in range(TICKS):
        # Movement at this offset happens before the tick, against these tiles.
        predicted_blocked = {
            pos for pos, until in timeline.bomb_until.items() if offset <= until
        }
        assert engine_bomb_tiles(world) == predicted_blocked, f"bombs @ {offset}"
        crates_now = {pos for pos in crate_cells(world.arena)}
        for pos, opens in timeline.crate_open.items():
            assert (pos in crates_now) == (offset < opens), f"crate {pos} @ {offset}"

        tick(world)

        predicted = timeline.lethal[offset] if offset < timeline.horizon else set[Pos]()
        assert engine_lethal(world) == predicted, f"lethal cells @ offset {offset}"
        for agent in agents:
            if agent.dead and (agent.x, agent.y) not in died_at:
                died_at[(agent.x, agent.y)] = offset

    assert died_at == {
        pos: offset for pos, offset in first_lethal.items() if offset is not None
    }


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_timeline_matches_engine(case: Case) -> None:
    world, agents = build_world(case)
    timeline = forecast(world)
    compare(world, agents, timeline)


@pytest.mark.parametrize("burned", [1, 2, 3])
def test_explosion_map_matches_engine(burned: int) -> None:
    """Observe mid-explosion: the map's forecast must match what kills next."""
    case = Case("burning", bombs=[((5, 5), 0)], crates=[(5, 6)], sitters=[(1, 1)])
    world, agents = build_world(case)
    for _ in range(burned):
        tick(world)
    # A sitter moved into the blast after the fact, as if it had walked in.
    agents[0].x, agents[0].y = 5, 4
    timeline = forecast(world)
    compare(world, agents, timeline)


def test_explosion_map_zero_is_harmless() -> None:
    """The final dangerous stage is observed as 0 and does not kill next step."""
    world, agents = build_world(Case("zero", bombs=[((5, 5), 0)], sitters=[(1, 1)]))
    tick(world)
    tick(world)
    agents[0].x, agents[0].y = 5, 4
    obs = observe(world, 0)
    assert obs.explosion_map[5, 4] == 0
    assert engine_lethal(world)  # still flagged dangerous in the engine right now
    tick(world)
    assert not agents[0].dead


def test_dropped_bomb_matches_engine() -> None:
    """The bomb this action drops is what ``own_bomb`` predicts."""
    world, agents = build_world(Case("drop", bombs=[], sitters=[(5, 5), (5, 8)]))
    timeline = forecast(world)
    geometry = Geometry(np.asarray(world.arena))
    timeline.add_bomb(geometry, crate_cells(world.arena), own_bomb((5, 5)))
    world.perform_agent_action(world.agents[0], "BOMB")
    compare(world, agents, timeline)


def test_blast_matches_engine_everywhere() -> None:
    """``Geometry.blast`` equals ``Bomb.get_blast_coords`` on every free cell."""
    field = arena(crates=[(3, 1), (1, 3), (5, 6)])
    geometry = Geometry(field)
    for x in range(17):
        for y in range(17):
            if field[x, y] == -1:
                continue
            bomb = cast(BombView, Bomb((x, y), None, 0, 3, None))
            engine = [(int(a), int(b)) for a, b in bomb.get_blast_coords(field)]
            assert list(geometry.blast((x, y))) == engine
