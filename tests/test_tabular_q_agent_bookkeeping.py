"""The shared callback collector works with opaque, non-tabular states."""

import pytest

from agent_code.tabular_q_agent.bookkeeping import (
    Bookkeeper,
    BookkeepingError,
    Transition,
)


def test_survived_and_terminal_transitions_keep_reward_ingredients() -> None:
    emitted: list[Transition[str]] = []
    keeper = Bookkeeper[str](emitted.append)
    keeper.begin_round(1)
    keeper.observe(1, 1, "first", (0, 5), 2, bomb_hits=3, crate_distance=4)
    keeper.chose(5, "BOMB")
    events = ["BOMB_DROPPED", "COIN_COLLECTED"]
    keeper.events_occurred(1, 1, "BOMB", events)
    events.append("GOT_KILLED")  # the engine mutates its own event list
    assert not emitted

    keeper.observe(1, 2, "second", (4,), None, crate_distance=1)
    first = emitted[0]
    assert first.observed.state == "first"
    assert first.observed.allowed == (0, 5)
    assert first.observed.coin_distance == 2
    assert first.observed.crate_distance == 4
    assert first.observed.bomb_hits == 3
    assert first.action == 5 and first.played == "BOMB"
    assert first.event_batches == (("BOMB_DROPPED", "COIN_COLLECTED"),)
    assert not first.terminal and first.next_observed is not None
    assert first.next_observed.state == "second"
    assert first.next_observed.allowed == (4,)
    assert first.next_observed.coin_distance is None
    assert first.next_observed.crate_distance == 1

    keeper.chose(4, "WAIT")
    keeper.events_occurred(1, 2, "WAIT", ["WAITED"])
    keeper.finish("WAIT", ["WAITED", "SURVIVED_ROUND"])
    last = emitted[1]
    assert last.observed.state == "second"
    assert last.terminal
    assert last.event_batches == (("WAITED",), ("SURVIVED_ROUND",))
    assert keeper.pending is None
    with pytest.raises(BookkeepingError):
        keeper.finish("WAIT", ["WAITED", "SURVIVED_ROUND"])
    assert len(emitted) == 2


def test_death_and_posthumous_events_form_one_terminal_transition() -> None:
    emitted: list[Transition[tuple[int, int]]] = []
    keeper = Bookkeeper[tuple[int, int]](emitted.append)
    keeper.begin_round(1)
    keeper.observe(1, 1, (2, 3), (5,), None, bomb_hits=2)
    keeper.chose(5, "BOMB")
    keeper.finish("BOMB", ["BOMB_DROPPED", "GOT_KILLED", "KILLED_OPPONENT"])
    (transition,) = emitted
    assert transition.terminal
    assert transition.observed.state == (2, 3)
    assert transition.event_batches == (
        ("BOMB_DROPPED", "GOT_KILLED", "KILLED_OPPONENT"),
    )
    keeper.begin_round(2)
    keeper.observe(2, 1, (1, 1), (4,), None)
    keeper.chose(4, "WAIT")
    keeper.finish("WAIT", ["WAITED", "SURVIVED_ROUND"])
    assert len(emitted) == 2
    assert emitted[1].observed.round == 2
