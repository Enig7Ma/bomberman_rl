"""Turn framework callbacks into completed transitions, independently of learning.

A surviving step completes at the next observation; the last or fatal step
completes at round end. The engine repeats a survivor's final events and delays
a dead agent's bomb credit. Keep the original event batches so a sink can
reproduce both the reward arithmetic and event counts without double counting.

States are opaque to this module. Callers must keep them immutable while they
are pending and sinks must copy mutable state data if they retain it in replay.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass


class BookkeepingError(RuntimeError):
    """The framework's callbacks arrived in an unexpected order."""


@dataclass(frozen=True)
class Observed[S]:
    round: int
    step: int
    state: S
    allowed: tuple[int, ...]
    coin_distance: int | None
    bomb_hits: int = 0
    crate_distance: int | None = None


@dataclass
class Pending[S]:
    observed: Observed[S]
    action: int
    played: str
    received: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Transition[S]:
    """One executed action, including reward ingredients and its successor.

    ``next_observed is None`` means terminal. Event batches contain each engine
    event once; a survivor's final batch contains only the newly added suffix.
    Distances and bomb hits remain available for reward recomputation in replay.
    """

    observed: Observed[S]
    action: int
    played: str
    event_batches: tuple[tuple[str, ...], ...]
    next_observed: Observed[S] | None

    @property
    def terminal(self) -> bool:
        return self.next_observed is None


class Bookkeeper[S]:
    def __init__(self, sink: Callable[[Transition[S]], None]) -> None:
        self._sink = sink
        self.round = 0
        self.pending: Pending[S] | None = None
        self._observed: Observed[S] | None = None

    def begin_round(self, round_number: int) -> None:
        if self.pending is not None:
            previous = self.pending.observed
            raise BookkeepingError(
                f"round {round_number} started while step {previous.step} of "
                f"round {previous.round} was still pending"
            )
        self.round = round_number
        self._observed = None

    def observe(
        self,
        round_number: int,
        step: int,
        state: S,
        allowed: Sequence[int],
        coin_distance: int | None,
        bomb_hits: int = 0,
        crate_distance: int | None = None,
    ) -> None:
        if round_number != self.round:
            raise BookkeepingError(
                f"observation of round {round_number} in round {self.round}"
            )
        observed = Observed(
            round_number,
            step,
            state,
            tuple(allowed),
            coin_distance,
            bomb_hits,
            crate_distance,
        )
        pending = self.pending
        if pending is not None:
            if pending.received is None:
                raise BookkeepingError(
                    f"step {step}: no game_events_occurred for step "
                    f"{pending.observed.step}"
                )
            self._emit(pending, (pending.received,), observed)
        self._observed = observed

    def chose(self, action: int, played: str) -> Pending[S]:
        observed = self._observed
        if observed is None:
            raise BookkeepingError("an action was chosen without an observation")
        self._observed = None
        self.pending = Pending(observed, action, played)
        return self.pending

    def events_occurred(
        self, round_number: int, step: int, played: str, events: Sequence[str]
    ) -> None:
        pending = self._require_pending("game_events_occurred")
        previous = pending.observed
        if (round_number, step) != (previous.round, previous.step):
            raise BookkeepingError(
                f"events for round {round_number} step {step}, but round "
                f"{previous.round} step {previous.step} is pending"
            )
        if played != pending.played:
            raise BookkeepingError(
                f"step {step}: the framework reports {played!r}, act returned "
                f"{pending.played!r}"
            )
        if pending.received is not None:
            raise BookkeepingError(f"step {step}: events reported twice")
        pending.received = tuple(events)

    def finish(self, played: str, events: Sequence[str]) -> None:
        pending = self._require_pending("end_of_round")
        if played != pending.played:
            raise BookkeepingError(
                f"end of round reports {played!r}, act returned {pending.played!r}"
            )
        delivered = tuple(events)
        seen = pending.received or ()
        if delivered[: len(seen)] != seen:
            raise BookkeepingError(
                f"end_of_round changed events already reported: {seen} -> {delivered}"
            )
        batches = (
            (delivered,)
            if pending.received is None
            else (pending.received, delivered[len(seen) :])
        )
        self._emit(pending, batches, None)

    def _require_pending(self, callback: str) -> Pending[S]:
        if self.pending is None:
            raise BookkeepingError(f"{callback} without a pending action")
        return self.pending

    def _emit(
        self,
        pending: Pending[S],
        event_batches: tuple[tuple[str, ...], ...],
        next_observed: Observed[S] | None,
    ) -> None:
        self._sink(
            Transition(
                pending.observed,
                pending.action,
                pending.played,
                event_batches,
                next_observed,
            )
        )
        self.pending = None
