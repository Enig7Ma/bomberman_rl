"""Result records produced by a tournament round."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentRoundResult:
    """Outcome for one agent in one round.

    ``seat`` is the agent's index in the lineup, which is also its index in the
    world's agent list and therefore the key used to pair a candidate against
    the control agent that played the same seat on the same seed.
    """

    code_name: str
    name: str
    seat: int
    score: int
    coins: int
    kills: int
    suicides: int
    crates: int
    bombs: int
    invalid: int
    moves: int
    steps: int
    survived: bool
    latency_mean: float
    latency_p99: float
    latency_max: float
    timeouts: int


@dataclass(frozen=True)
class RoundResult:
    """Outcome of one round, for every agent that played it."""

    scenario: str
    seed: int
    lineup: tuple[str, ...]
    arm: str
    focus_seat: int
    steps: int
    agents: tuple[AgentRoundResult, ...]

    @property
    def focus(self) -> AgentRoundResult:
        """The agent this round was scheduled to measure.

        Looked up by ``seat`` rather than by position, so the record stays
        correct if results are ever stored in a different order.
        """
        for agent in self.agents:
            if agent.seat == self.focus_seat:
                return agent
        raise ValueError(f"no agent in seat {self.focus_seat} of {self.lineup}")
