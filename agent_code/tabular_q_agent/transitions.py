"""Tabular sink for the shared callback bookkeeping (plan §5.8).

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
bootstrap). ``Bookkeeper`` copies events as they arrive and delivers only
the part of the final list that was not reported before -- after checking that the
reported part is unchanged, so a framework change fails loudly instead of
double-counting quietly.

``Trainer`` consumes each completed transition, counts its event batches,
applies the tabular update and builds the round's metrics. Its public callback
methods delegate to ``Bookkeeper`` so the existing agent API stays unchanged.

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

import events as e

from .bookkeeping import Bookkeeper, Pending, Transition
from .bookkeeping import BookkeepingError as BookkeepingError
from .learner import Learner, Selection
from .metrics import RoundRecord
from .rewards import Rewards


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
        self.bookkeeper = Bookkeeper[int](self._on_transition)
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

    @property
    def pending(self) -> Pending[int] | None:
        return self.bookkeeper.pending

    def begin_round(self, round_number: int, opponents: Sequence[str]) -> None:
        self.bookkeeper.begin_round(round_number)
        self._reset(round_number, opponents)

    def observe(
        self,
        round_number: int,
        step: int,
        state: int,
        allowed: Sequence[int],
        coin_distance: int | None,
        bomb_hits: int = 0,
        crate_distance: int | None = None,
    ) -> None:
        self.bookkeeper.observe(
            round_number,
            step,
            state,
            allowed,
            coin_distance,
            bomb_hits,
            crate_distance,
        )

    def chose(self, selection: Selection, played: str) -> None:
        pending = self.bookkeeper.chose(selection.action, played)
        self._steps += 1
        self._forced += len(pending.observed.allowed) == 1
        self._explored += selection.explored
        self._unseen += selection.unseen and not selection.explored

    def events_occurred(
        self, round_number: int, step: int, played: str, events: Sequence[str]
    ) -> None:
        self.bookkeeper.events_occurred(round_number, step, played, events)

    def finish(self, played: str, events: Sequence[str]) -> RoundRecord:
        self.bookkeeper.finish(played, events)

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

    def _count(self, bomb_hits: int, events: Sequence[str]) -> float:
        """Reward for newly delivered events; each event is delivered once."""
        self._events.update(events)
        base = self.rewards.base(events)
        self._base += base
        aid = self.rewards.aids(events)
        if e.BOMB_DROPPED in events:
            aid += self.rewards.bomb(bomb_hits)
        return base + aid

    def _on_transition(self, transition: Transition[int]) -> None:
        observed = transition.observed
        reward = 0.0
        for batch in transition.event_batches:
            reward += self._count(observed.bomb_hits, batch)
        phi = self.rewards.potential(observed.coin_distance, observed.crate_distance)
        following = transition.next_observed
        phi_next = (
            0.0
            if following is None
            else self.rewards.potential(
                following.coin_distance, following.crate_distance
            )
        )
        reward += self.rewards.shaping(phi, phi_next)
        self._alpha_sum += self.learner.step_size(observed.state, transition.action)
        delta = self.learner.update(
            observed.state,
            transition.action,
            reward,
            following.state if following is not None else None,
            following.allowed if following is not None else (),
        )
        self._td_sum += abs(delta)
        self._return += reward
        self._updates += 1
