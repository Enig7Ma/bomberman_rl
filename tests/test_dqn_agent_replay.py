"""Replay ownership, FIFO replacement and reward recomputation."""

from dataclasses import fields, replace
from typing import Any, cast

import numpy as np
import pytest

import events as e
from agent_code.dqn_agent.replay import ReplayBuffer, ReplayTransition
from agent_code.dqn_agent.rewards import Rewards


def transition(index: int = 0, /, **changes: object) -> ReplayTransition:
    return replace(
        ReplayTransition(
            x=np.array([index, 1], dtype=np.float32),
            a=1,
            base=1.0,
            crates=2,
            deaths=0,
            phi_unit=0.5,
            phi_unit_next=0.25,
            x_next=np.array([index + 1, 2], dtype=np.float32),
            mask_next=np.array([False, True, False, False, True, False]),
            done=False,
            stage=0,
            round_id=1,
            transition_id=index,
        ),
        **changes,
    )


def test_fifo_len_and_multiple_wraps() -> None:
    replay = ReplayBuffer(3, 2)
    assert len(replay) == 0
    for index in range(11):
        replay.push(transition(index))
        assert len(replay) == min(index + 1, 3)
        batch = replay.sample(256, np.random.default_rng(index), gamma=0.99)
        expected = set(range(max(0, index - 2), index + 1))
        assert set(batch.transition_id.tolist()) == expected
        np.testing.assert_array_equal(batch.x[:, 0], batch.transition_id)
        np.testing.assert_array_equal(batch.x_next[:, 0], batch.transition_id + 1)


def test_sampling_is_uniform_seeded_and_with_replacement() -> None:
    replay = ReplayBuffer(100, 2)
    for i in range(4):
        replay.push(transition(i))
    before = np.random.get_state()
    first = replay.sample(10_000, np.random.default_rng(7), gamma=0.99)
    second = replay.sample(10_000, np.random.default_rng(7), gamma=0.99)
    for field in fields(first):
        np.testing.assert_array_equal(
            getattr(first, field.name), getattr(second, field.name)
        )
    counts = np.bincount(first.transition_id.astype(np.int64))
    assert len(counts) == 4 and np.all((counts > 2200) & (counts < 2800))
    other = replay.sample(10_000, np.random.default_rng(8), gamma=0.99)
    assert not np.array_equal(first.transition_id, other.transition_id)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])


def test_input_and_batch_arrays_do_not_alias_storage() -> None:
    replay = ReplayBuffer(1, 2)
    t = transition(3)
    replay.push(t)
    original = replay.sample(1, np.random.default_rng(0), gamma=0.99)
    t.x.fill(99)
    t.x_next.fill(88)
    t.mask_next.fill(False)
    batch = replay.sample(1, np.random.default_rng(0), gamma=0.99)
    for field in fields(batch):
        np.testing.assert_array_equal(
            getattr(batch, field.name), getattr(original, field.name)
        )
        getattr(batch, field.name).fill(0)
    again = replay.sample(1, np.random.default_rng(0), gamma=0.99)
    for field in fields(again):
        np.testing.assert_array_equal(
            getattr(again, field.name), getattr(original, field.name)
        )
    replay.push(transition(4))
    assert original.transition_id.tolist() == [3]
    assert replay.sample(
        1, np.random.default_rng(0), gamma=0.99
    ).transition_id.tolist() == [4]


def test_terminal_normalisation_and_exact_storage_dtypes() -> None:
    replay = ReplayBuffer(1, 2)
    replay.push(
        transition(
            done=True, deaths=1, stage=255, round_id=2**32 - 1, transition_id=2**64 - 1
        )
    )
    batch = replay.sample(2, np.random.default_rng(0), gamma=0.99, c_coin=2)
    assert batch.done.all()
    assert not batch.mask_next.any() and not batch.x_next.any()
    assert not batch.phi_unit_next.any()
    assert batch.r.tolist() == [0, 0]  # 1 + 2 * (0 - 0.5)
    assert batch.stage.tolist() == [255, 255]
    assert batch.round_id.tolist() == [2**32 - 1] * 2
    assert batch.transition_id.tolist() == [2**64 - 1] * 2
    for name in ("x", "x_next", "base", "phi_unit", "phi_unit_next", "r"):
        assert getattr(batch, name).dtype == np.float32
    for name in ("a", "crates", "deaths", "stage"):
        assert getattr(batch, name).dtype == np.uint8
    assert batch.round_id.dtype == np.uint32
    assert batch.transition_id.dtype == np.uint64
    assert batch.mask_next.dtype == batch.done.dtype == np.bool_
    for field in fields(batch):
        assert getattr(batch, field.name).flags.c_contiguous
    assert batch.x.shape == batch.x_next.shape == (2, 2)
    assert batch.mask_next.shape == (2, 6)


@pytest.mark.parametrize("gamma", [0.0, 0.99, 1.0])
def test_reward_components_match_rewards_after_coefficient_changes(
    gamma: float,
) -> None:
    rng = np.random.default_rng(72)
    replay = ReplayBuffer(100, 2)
    inputs: list[tuple[list[str], int | None, int | None, bool]] = []
    for i in range(100):
        crates = int(rng.integers(8))
        dead = bool(rng.integers(2))
        terminal = dead or i % 3 == 0  # alive at round cap is also terminal
        events = (
            [e.COIN_COLLECTED] * int(rng.integers(4))
            + [e.KILLED_OPPONENT] * int(rng.integers(4))
            + [e.CRATE_DESTROYED] * crates
            + ([e.GOT_KILLED, e.KILLED_SELF] if dead else [])
            + [e.INVALID_ACTION, e.WAITED, e.BOMB_DROPPED]
        )
        distance = None if i % 5 == 0 else int(rng.integers(25))
        next_distance = None if i % 7 == 0 else int(rng.integers(25))
        unit = Rewards(gamma, coin_potential=1)
        replay.push(
            transition(
                i,
                base=unit.base(events),
                crates=crates,
                deaths=int(dead),
                phi_unit=unit.potential(distance),
                phi_unit_next=unit.potential(next_distance),
                done=terminal,
            )
        )
        inputs.append((events, distance, next_distance, terminal))
    for coin, crate, death in ((0, 0, 0), (0.5, 0, 0), (1.5, 0.25, -2), (0, -0.5, -1)):
        rewards = Rewards(gamma, coin_potential=coin, crate_aid=crate, death_aid=death)
        batch = replay.sample(
            500,
            np.random.default_rng(6),
            gamma=gamma,
            c_coin=coin,
            crate_aid=crate,
            death_aid=death,
        )
        expected: list[float] = []
        for index in batch.transition_id:
            events, d, next_d, terminal = inputs[int(index)]
            expected.append(
                rewards.base(events)
                + rewards.aids(events)
                + rewards.shaping(
                    rewards.potential(d), 0.0 if terminal else rewards.potential(next_d)
                )
            )
        np.testing.assert_allclose(batch.r, expected, atol=2e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "capacity,dim", [(0, 2), (-1, 2), (True, 2), (1.5, 2), (2, 0), (2, -1)]
)
def test_invalid_dimensions(capacity: object, dim: object) -> None:
    with pytest.raises(ValueError):
        ReplayBuffer(cast(int, capacity), cast(int, dim))


@pytest.mark.parametrize(
    "name,value",
    [
        ("a", -1),
        ("a", 6),
        ("a", True),
        ("a", 1.5),
        ("crates", 256),
        ("crates", -1),
        ("deaths", 2),
        ("stage", 256),
        ("round_id", 2**32),
        ("transition_id", 2**64),
        ("transition_id", -1),
        ("done", 1),
        ("base", float("nan")),
        ("base", float("inf")),
        ("base", 1e40),
        ("base", "bad"),
        ("phi_unit", -0.1),
        ("phi_unit_next", 1.1),
        ("phi_unit_next", float("nan")),
        ("x", np.zeros(3, dtype=np.float32)),
        ("x_next", np.zeros((1, 2), dtype=np.float32)),
        ("x", np.zeros(2)),
        ("x_next", np.array([0, np.inf], dtype=np.float32)),
        ("x", [0, 1]),
        ("mask_next", np.ones(6, dtype=np.uint8)),
        ("mask_next", np.ones(5, dtype=bool)),
        ("mask_next", np.zeros(6, dtype=bool)),
    ],
)
def test_invalid_push_preserves_existing_row_and_write_position(
    name: str, value: object
) -> None:
    replay = ReplayBuffer(2, 2)
    replay.push(transition(7))
    with pytest.raises(ValueError):
        replay.push(transition(**{name: value}))
    assert len(replay) == 1
    assert replay.sample(
        1, np.random.default_rng(0), gamma=0.99
    ).transition_id.tolist() == [7]

    replay.push(transition(8))
    replay.push(transition(9))
    assert set(
        replay.sample(128, np.random.default_rng(0), gamma=0.99).transition_id.tolist()
    ) == {8, 9}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"batch_size": -1},
        {"batch_size": True},
        {"gamma": -0.1},
        {"gamma": 1.1},
        {"gamma": float("nan")},
        {"c_coin": float("inf")},
        {"crate_aid": True},
        {"death_aid": "bad"},
        {"rng": None},
        {"crate_aid": 3e38},
    ],
)
def test_invalid_sample(kwargs: dict[str, Any]) -> None:
    replay = ReplayBuffer(1, 2)
    replay.push(transition())
    params: dict[str, Any] = {
        "batch_size": 1,
        "rng": np.random.default_rng(0),
        "gamma": 0.99,
        **kwargs,
    }
    with pytest.raises(ValueError):
        replay.sample(**params)


def test_empty_buffer() -> None:
    with pytest.raises(ValueError, match="empty"):
        ReplayBuffer(2, 2).sample(1, np.random.default_rng(0), gamma=0.99)
