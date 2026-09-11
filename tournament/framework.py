"""Typed views over the untyped game framework.

``agents.py`` and ``environment.py`` predate this repository's type-checking
setup and are excluded from it, so every attribute they expose arrives as
``Unknown``. Rather than scatter casts through the harness, each attribute the
tournament actually reads is declared once here and the framework objects are
cast to these protocols at the boundary.

Nothing else in ``tournament/`` should reach into framework internals directly.
"""

from collections import defaultdict
from collections.abc import Callable
from typing import Protocol, cast

from environment import BombeRLeWorld


class TimedAgent(Protocol):
    """The little of an agent that latency recording needs."""

    available_think_time: float | None

    # Declared as an attribute rather than a method so the latency
    # instrumentation can replace it per instance.
    wait_for_act: Callable[[], tuple[str, float]]


class AgentView(TimedAgent, Protocol):
    """The subset of ``agents.Agent`` the tournament reads.

    ``score``, ``dead`` and ``statistics`` are None until the agent's first
    round starts -- the framework initialises them in ``start_round``.
    """

    name: str
    code_name: str
    x: int
    y: int
    score: int | None
    dead: bool | None
    statistics: defaultdict[str, int] | None


def agents_of(world: BombeRLeWorld) -> list[AgentView]:
    """The world's agents, in seat order, as typed views."""
    return [cast(AgentView, agent) for agent in world.agents]
