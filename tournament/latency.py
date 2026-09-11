"""Per-step decision-latency instrumentation.

The framework already accumulates think time into ``Agent.statistics['time']``,
but a running sum only yields a mean. ``settings.TIMEOUT`` is a *per-step* cap,
so what decides whether an agent is legal is the tail, not the average.

Wrapping ``Agent.wait_for_act`` records every individual measurement without
modifying framework code. The timeout test mirrors ``poll_and_run_agents``:
an action is discarded when its think time exceeds the budget *remaining for
that step*, which shrinks after a previous overrun.
"""

import math
from dataclasses import dataclass, field

from environment import BombeRLeWorld
from tournament.framework import TimedAgent, agents_of


@dataclass
class LatencyRecord:
    """Every decision latency measured for one agent in one round."""

    samples: list[float] = field(default_factory=list[float])
    timeouts: int = 0

    def add(self, seconds: float, *, timed_out: bool) -> None:
        self.samples.append(seconds)
        if timed_out:
            self.timeouts += 1

    @property
    def mean(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else 0.0

    @property
    def maximum(self) -> float:
        return max(self.samples, default=0.0)

    def percentile(self, q: float) -> float:
        """Nearest-rank percentile, so the value is always an observed sample."""
        if not self.samples:
            return 0.0
        ordered = sorted(self.samples)
        rank = max(1, math.ceil(q / 100.0 * len(ordered)))
        return ordered[rank - 1]


def instrument(agent: TimedAgent) -> LatencyRecord:
    """Start recording one agent's decision latencies."""
    record = LatencyRecord()
    original = agent.wait_for_act

    def wrapped() -> tuple[str, float]:
        budget = agent.available_think_time
        action, think_time = original()
        record.add(think_time, timed_out=budget is not None and think_time > budget)
        return action, think_time

    agent.wait_for_act = wrapped
    return record


def instrument_world(world: BombeRLeWorld) -> list[LatencyRecord]:
    """Instrument every agent in the world, in seat order."""
    return [instrument(agent) for agent in agents_of(world)]
