"""Lossless E3 inputs, schema compatibility and canonical consistency."""

import random
from dataclasses import replace
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from agent_code.dqn_agent import encoder, features
from agent_code.dqn_agent.encoder import ENCODERS, Encoder, OneHotE3
from agent_code.dqn_agent.features import (
    ENCODINGS,
    RADIX,
    Extracted,
    Features,
    FieldName,
)
from agent_code.dqn_agent.symmetry import (
    SYMMETRIES,
    Symmetry,
    canonical,
    transform_features,
)

E3 = ENCODINGS["E3"]


def test_layout_and_protocol() -> None:
    subject: Encoder = OneHotE3()
    assert subject.name == "onehot_e3" and subject.dim == 32
    assert ENCODERS[subject.name].schema_id == subject.schema_id
    state = Features(
        mask=37, coin_dir=1, crate_dir=5, bomb_yield=3, danger=1, opp_dir=4, attack=2
    )
    actual = subject.encode(state)
    expected = np.zeros(32, dtype=np.float32)
    # Independent offsets: mask 0:6; coin 6:12; crate 12:18; yield 18:22;
    # danger 22; opponent 23:29; attack 29:32.
    expected[[0, 2, 5, 7, 17, 21, 22, 27, 31]] = 1.0
    assert actual.shape == (32,) and actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)


def test_ten_thousand_distinct_states_round_trip_without_collisions() -> None:
    subject = OneHotE3()
    indices = random.Random(20260916).sample(range(E3.n_states), 10_000)
    seen: set[bytes] = set()
    for index in indices:
        state = E3.decode(index)
        vector = subject.encode(state)
        assert subject.decode(vector) == state
        for start, end in ((6, 12), (12, 18), (18, 22), (23, 29), (29, 32)):
            assert np.count_nonzero(vector[start:end]) == 1
        seen.add(vector.tobytes())
    assert len(seen) == len(indices)


@pytest.mark.parametrize("mask", range(64))
@pytest.mark.parametrize("danger", [0, 1])
def test_bits_are_lossless_including_zero_and_full_masks(
    mask: int, danger: int
) -> None:
    subject = OneHotE3()
    state = Features(mask=mask, danger=danger)
    vector = subject.encode(state)
    assert vector[:6].tolist() == [(mask >> bit) & 1 for bit in range(6)]
    assert vector[22] == danger
    assert subject.decode(vector) == state


def test_encode_into_reuses_a_buffer_and_clears_previous_categories() -> None:
    subject = OneHotE3()
    replay_row = np.full(34, -7.0, dtype=np.float32)
    out = replay_row[1:33]
    address = out.ctypes.data
    for state in (E3.decode(E3.n_states - 1), E3.decode(0), Features()):
        assert subject.encode_into(out, state) is None
        np.testing.assert_array_equal(out, subject.encode(state))
        assert out.ctypes.data == address
        assert replay_row[0] == replay_row[-1] == -7.0


def test_extracted_information_is_not_part_of_onehot_e3() -> None:
    subject = OneHotE3()
    state = Features(mask=17, coin_dir=0)
    extracted = Extracted(
        features=Features(),
        allowed=("BOMB",),
        best_tier=4,
        coin_distance=100,
        crate_distance=30,
        bomb_hits=50,
        options={},
    )
    np.testing.assert_array_equal(
        subject.encode(state, extracted), subject.encode(state)
    )


@pytest.mark.parametrize("symmetry", SYMMETRIES)
def test_all_symmetries_encode_the_same_canonical_features(symmetry: Symmetry) -> None:
    subject = OneHotE3()
    # Covers both trivial and non-trivial stabilisers. The vector is invariant
    # even when the action-frame transform is not uniquely determined.
    states = [Features(), E3.decode(0), E3.decode(E3.n_states - 1)]
    states += [E3.decode(i) for i in random.Random(42).sample(range(E3.n_states), 256)]
    for state in states:
        index, _ = canonical(state, E3)
        image_index, _ = canonical(transform_features(state, symmetry), E3)
        np.testing.assert_array_equal(
            subject.encode(E3.decode(index)), subject.encode(E3.decode(image_index))
        )


def test_schema_tracks_feature_version(monkeypatch: pytest.MonkeyPatch) -> None:
    subject = OneHotE3()
    original = subject.schema_id
    assert len(original) == 16 and OneHotE3().schema_id == original
    monkeypatch.setattr(features, "FEATURE_VERSION", features.FEATURE_VERSION + 1)
    assert subject.schema_id != original


def test_schema_tracks_block_order_even_with_identical_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subject = OneHotE3()
    original = subject.schema_id
    blocks = list(encoder.BLOCKS)
    blocks[1], blocks[2] = blocks[2], blocks[1]
    monkeypatch.setattr(encoder, "BLOCKS", tuple(blocks))
    assert subject.dim == 32
    assert subject.schema_id != original
    state = Features(coin_dir=1, crate_dir=5)
    assert subject.decode(subject.encode(state)) == state


@pytest.mark.parametrize("field", list(RADIX))
def test_out_of_range_categories_are_rejected(field: FieldName) -> None:
    subject = OneHotE3()
    for value in (-1, RADIX[field], 0.5):
        with pytest.raises(ValueError, match=field):
            subject.encode(replace(Features(), **{field: value}))


@pytest.mark.parametrize("shape", [(31,), (33,), (1, 32)])
def test_wrong_vector_shape_is_rejected(shape: tuple[int, ...]) -> None:
    subject = OneHotE3()
    vector = np.zeros(shape, dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        subject.encode_into(vector, Features())
    with pytest.raises(ValueError, match="shape"):
        subject.decode(vector)


def test_wrong_dtype_is_rejected() -> None:
    vector = cast(NDArray[np.float32], np.zeros(32, dtype=np.float64))
    with pytest.raises(ValueError, match="float32"):
        OneHotE3().encode_into(vector, Features())
    with pytest.raises(ValueError, match="float32"):
        OneHotE3().decode(vector)


@pytest.mark.parametrize("bad", [0.5, -1.0, 2.0, float("nan"), float("inf")])
def test_decode_rejects_non_binary_values(bad: float) -> None:
    subject = OneHotE3()
    vector = subject.encode(Features())
    vector[0] = bad
    with pytest.raises(ValueError, match="binary"):
        subject.decode(vector)


@pytest.mark.parametrize("start,end", [(6, 12), (12, 18), (18, 22), (23, 29), (29, 32)])
@pytest.mark.parametrize("active", [0, 2])
def test_decode_rejects_invalid_onehot_blocks(
    start: int, end: int, active: int
) -> None:
    subject = OneHotE3()
    vector = subject.encode(Features())
    vector[start:end] = 0
    vector[start : start + active] = 1
    with pytest.raises(ValueError, match="exactly one"):
        subject.decode(vector)
