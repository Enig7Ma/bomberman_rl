"""From the framework's callbacks to Q-learning transitions (plan §5.8).

The engine reports the consequences of a step in two different ways (plan §4,
facts E1-E4):

- a *surviving* agent gets ``game_events_occurred`` after every step,
  including the last; on the last step ``end_of_round`` then repeats that
  step's events, in the same list, with ``SURVIVED_ROUND`` appended;
- an agent that *died* gets no ``game_events_occurred`` for its death step;
  ``end_of_round`` delivers the death step's events, plus whatever its bombs
  did afterwards if the round went on.

So neither callback can complete a transition on its own. Each action leaves
a transition *pending* until either the next ``act`` (the step was survived:
non-terminal, bootstrap from the new state) or ``end_of_round`` (terminal, no
bootstrap). Events are counted as they arrive, and ``end_of_round`` counts only
the part of its list that was not reported before -- after checking that the
reported part is unchanged, so a framework change fails loudly instead of
double-counting quietly.

Every action gets exactly one update. Its reward is the event reward (score
delta plus aids, see ``rewards``) plus ``gamma * phi(s') - phi(s)``, with
``phi = 0`` after a terminal transition. Posthumous events are folded into the
death transition undiscounted: the score total stays exact, the timing does
not.

Any callback arriving out of the expected order raises ``BookkeepingError``;
a silently wrong target is worse than a crashed training run.
"""

import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import events as e

from .learner import Learner, Selection
from .metrics import RoundRecord
from .rewards import Rewards


class BookkeepingError(RuntimeError):
    """The framework's callbacks arrived in an order this module does not expect."""


@dataclass
class Pending:
    """An action whose consequences are not complete yet."""

    round: int
    step: int
    state: int
    action: int
    played: str
    phi: float
    reward: float = 0.0
    # The events ``game_events_occurred`` reported, or None if it never came.
    received: tuple[str, ...] | None = None


@dataclass(frozen=True)
class _Observed:
    round: int
    step: int
    state: int
    allowed: tuple[int, ...]
    phi: float


class Trainer:
    def __init__(
        self, learner: Learner, rewards: Rewards, *, epsilon: float, stage: str = ""
    ) -> None:
        self.learner = learner
        self.rewards = rewards
        self.epsilon = epsilon
        self.stage = stage
        meta = learner.table.meta
        self.rounds_trained = int(meta.get("rounds_trained", 0))
        self.steps_trained = int(meta.get("steps_trained", 0))
        self.pending: Pending | None = None
        self._observed: _Observed | None = None
        self._reset(0, ())

    def _reset(self, round_number: int, opponents: Sequence[str]) -> None:
        self.round = round_number
        self._opponents = list(opponents)
        self._events: Counter[str] = Counter()
        self._steps = 0
        self._updates = 0
        self._forced = 0
        self._explored = 0
        self._unseen = 0
        self._alpha_sum = 0.0
        self._td_sum = 0.0
        self._base = 0.0
        self._return = 0.0
        self._visited_at_start = self.learner.table.visited_states
        self._started = time.perf_counter()

    # --- called from ``act`` ----------------------------------------------

    def begin_round(self, round_number: int, opponents: Sequence[str]) -> None:
        if self.pending is not None:
            raise BookkeepingError(
                f"round {round_number} started while step {self.pending.step} of "
                f"round {self.pending.round} was still pending"
            )
        self._reset(round_number, opponents)

    def observe(
        self,
        round_number: int,
        step: int,
        state: int,
        allowed: Sequence[int],
        coin_distance: int | None,
    ) -> None:
        """A new observation; completes the pending transition as survived."""
        if round_number != self.round:
            raise BookkeepingError(
                f"observation of round {round_number} in round {self.round}"
            )
        phi = self.rewards.potential(coin_distance)
        pending = self.pending
        if pending is not None:
            if pending.received is None:
                raise BookkeepingError(
                    f"step {step}: no game_events_occurred for step {pending.step}"
                )
            shaping = self.rewards.shaping(pending.phi, phi)
            self._update(pending, pending.reward + shaping, state, allowed)
            self.pending = None
        self._observed = _Observed(round_number, step, state, tuple(allowed), phi)

    def chose(self, selection: Selection, played: str) -> None:
        """The action ``act`` returns for the last observation."""
        observed = self._observed
        if observed is None:
            raise BookkeepingError("an action was chosen without an observation")
        self._observed = None
        self.pending = Pending(
            observed.round,
            observed.step,
            observed.state,
            selection.action,
            played,
            observed.phi,
        )
        self._steps += 1
        self._forced += len(observed.allowed) == 1
        self._explored += selection.explored
        self._unseen += selection.unseen and not selection.explored

    # --- called from ``train`` --------------------------------------------

    def events_occurred(
        self, round_number: int, step: int, played: str, events: Sequence[str]
    ) -> None:
        """``game_events_occurred``: the pending step was survived."""
        pending = self._require_pending("game_events_occurred")
        if (round_number, step) != (pending.round, pending.step):
            raise BookkeepingError(
                f"events for round {round_number} step {step}, but round "
                f"{pending.round} step {pending.step} is pending"
            )
        if played != pending.played:
            raise BookkeepingError(
                f"step {step}: the framework reports {played!r}, act returned "
                f"{pending.played!r}"
            )
        if pending.received is not None:
            raise BookkeepingError(f"step {step}: events reported twice")
        received = tuple(events)  # the engine keeps mutating its list
        pending.received = received
        pending.reward += self._count(received)

    def finish(self, played: str, events: Sequence[str]) -> RoundRecord:
        """``end_of_round``: complete the last transition as terminal."""
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
        pending.reward += self._count(delivered[len(seen) :])
        shaping = self.rewards.shaping(pending.phi, 0.0)
        self._update(pending, pending.reward + shaping, None, ())
        self.pending = None

        self.rounds_trained += 1
        self.steps_trained += self._steps
        table = self.learner.table
        table.meta["rounds_trained"] = self.rounds_trained
        table.meta["steps_trained"] = self.steps_trained
        visited = table.visited_states
        counts = self._events
        return RoundRecord(
            round=self.round,
            rounds_trained=self.rounds_trained,
            stage=self.stage,
            opponents=self._opponents,
            steps=self._steps,
            updates=self._updates,
            survived=counts[e.SURVIVED_ROUND] > 0,
            died=counts[e.GOT_KILLED] > 0,
            self_kill=counts[e.KILLED_SELF] > 0,
            base_reward=self._base,
            shaped_return=self._return,
            coins=counts[e.COIN_COLLECTED],
            kills=counts[e.KILLED_OPPONENT],
            crates=counts[e.CRATE_DESTROYED],
            bombs=counts[e.BOMB_DROPPED],
            invalid=counts[e.INVALID_ACTION],
            event_counts=dict(sorted(counts.items())),
            epsilon=self.epsilon,
            mean_alpha=self._alpha_sum / self._updates,
            mean_abs_td=self._td_sum / self._updates,
            visited_states=visited,
            new_states=visited - self._visited_at_start,
            forced_fraction=self._forced / self._steps,
            explored_steps=self._explored,
            unseen_decisions=self._unseen,
            wall_time=time.perf_counter() - self._started,
        )

    # --- internals ----------------------------------------------------------

    def _require_pending(self, callback: str) -> Pending:
        if self.pending is None:
            raise BookkeepingError(f"{callback} without a pending action")
        return self.pending

    def _count(self, events: Sequence[str]) -> float:
        self._events.update(events)
        base = self.rewards.base(events)
        self._base += base
        return base + self.rewards.aids(events)

    def _update(
        self,
        pending: Pending,
        reward: float,
        next_state: int | None,
        next_allowed: Sequence[int],
    ) -> None:
        self._alpha_sum += self.learner.step_size(pending.state, pending.action)
        delta = self.learner.update(
            pending.state, pending.action, reward, next_state, next_allowed
        )
        self._td_sum += abs(delta)
        self._return += reward
        self._updates += 1
