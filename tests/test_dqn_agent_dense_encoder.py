"""The dense_v1 input: layout, ranges, the one-hot prefix and canonical frames.

The one-hot encoder is tested in ``test_dqn_agent_encoder.py``; this file is
about the blocks dense_v1 adds, and above all about the property that makes
them safe to learn from: two boards that are images of each other under a
board symmetry must produce the *same* canonical input vector, because their
canonical action frames are the same.
"""

import random
from typing import Any, cast

import numpy as np
import pytest

from agent_code.dqn_agent.core.world_model import ACTIONS, Observation
from agent_code.dqn_agent.encoder import (
    DISTANCE_FIELDS,
    FIXED_FEATURES,
    MOVE_FEATURES,
    RACE_SCALARS,
    SCALARS,
    DenseV1,
    DenseV2,
    DenseV3,
    Encoder,
    OneHotE3,
)
from agent_code.dqn_agent.features import ENCODINGS, Extracted, Extractor, Features
from agent_code.dqn_agent.symmetry import (
    IDENTITY,
    SYMMETRIES,
    Symmetry,
    canonical,
    transform_features,
    transform_game_state,
)
from tournament.engine import (
    WorldConfig,
    create_world,
    quiet_logging,
    reset_framework_logging,
)

E3 = ENCODINGS["E3"]
PREFIX = OneHotE3().dim
# ``features.NONE``: the direction fields' "there is none" value.
NONE_DIRECTION = 4


def extracted(**changes: object) -> Extracted:
    base: dict[str, object] = {
        "features": Features(),
        "allowed": ("UP", "WAIT"),
        "best_tier": 4,
        "coin_distance": None,
        "crate_distance": None,
        "bomb_hits": 0,
        "options": {},
    }
    return Extracted(**cast(Any, {**base, **changes}))


def test_layout_and_protocol() -> None:
    subject: Encoder = DenseV1()
    assert subject.name == "dense_v1"
    assert subject.dim == PREFIX + 4 * MOVE_FEATURES + 2 * FIXED_FEATURES + 2 * len(
        DISTANCE_FIELDS
    ) + len(SCALARS)
    assert subject.schema_id != OneHotE3().schema_id


def test_prefix_is_exactly_the_onehot_encoding() -> None:
    state = Features(
        mask=37, coin_dir=1, crate_dir=5, bomb_yield=3, danger=1, opp_dir=4, attack=2
    )
    dense = DenseV1().encode(state, extracted())
    np.testing.assert_array_equal(dense[:PREFIX], OneHotE3().encode(state))


def test_every_value_is_a_unit_interval() -> None:
    """Nothing in the input may dwarf the one-hot bits: the network starts
    with one scale for all of them."""
    rng = random.Random(3)
    for _ in range(200):
        sample = extracted(
            tiers=tuple(rng.randrange(-1, 5) for _ in ACTIONS),
            refuges=tuple(rng.randrange(0, 60) for _ in ACTIONS),
            coin_distance=rng.choice([None, 0, 3, 40]),
            crate_distance=rng.choice([None, 1, 25]),
            opponent_distance=rng.choice([None, 2, 90]),
            hunt_distance=rng.choice([None, 4]),
            lethal_offsets=rng.choice([(), (0,), (2, 3), (7,)]),
            bomb_hits=rng.randrange(0, 20),
            coins_visible=rng.randrange(0, 60),
            opponents_alive=rng.randrange(0, 4),
            crates_left=rng.randrange(0, 300),
            step=rng.randrange(0, 401),
            options={"coin_dir": frozenset({rng.randrange(4)})},
        )
        vector = DenseV1().encode(Features(mask=63), sample)
        assert vector.dtype == np.float32
        assert np.isfinite(vector).all()
        assert vector.min() >= 0.0 and vector.max() <= 1.0


def test_per_move_blocks_follow_the_canonical_frame() -> None:
    """A tier that belongs to raw UP must be written in the slot of UP's image,
    because the network's outputs are in that same canonical frame."""
    sample = extracted(
        tiers=(4, 0, 1, 2, 3, -1),
        refuges=(8, 0, 1, 2, 0, 0),
        options={"coin_dir": frozenset({0})},
    )
    for symmetry in SYMMETRIES:
        vector = DenseV1().encode(Features(mask=63), sample, symmetry)
        block = PREFIX + symmetry.direction(0) * MOVE_FEATURES
        assert vector[block] == 1.0  # UP is legal
        assert vector[block + 1] == pytest.approx(1.0)  # tier 4 / 4
        assert vector[block + 2] == pytest.approx(1.0)  # 8 refuges
        assert vector[block + 3] == 1.0  # the coin lies that way
        # BOMB is not legal here and keeps its own, symmetry-fixed slot.
        bomb = PREFIX + 4 * MOVE_FEATURES + FIXED_FEATURES
        assert vector[bomb] == 0.0


def test_missing_distances_are_marked_and_not_silently_zero() -> None:
    """Zero is "right here", which is the opposite of "there is none"."""
    near = DenseV1().encode(Features(), extracted(coin_distance=0))
    none = DenseV1().encode(Features(), extracted(coin_distance=None))
    offset = PREFIX + 4 * MOVE_FEATURES + 2 * FIXED_FEATURES
    assert (near[offset], near[offset + 1]) == (0.0, 0.0)
    assert (none[offset], none[offset + 1]) == (1.0, 1.0)


def test_extracted_is_required() -> None:
    with pytest.raises(ValueError, match="Extracted"):
        DenseV1().encode(Features())


@pytest.fixture(scope="module")
def engine_states() -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    with quiet_logging():
        for scenario, seed in (("classic", 0), ("classic", 1), ("loot-crate", 2)):
            world: Any = create_world(WorldConfig(("bfs_agent",) * 4, scenario, seed))
            try:
                world.new_round()
                world.user_input = None  # normally set by do_step
                while world.running:
                    if world.step % 3 == 0:
                        for agent in world.active_agents:
                            states.append(world.get_state_for_agent(agent))
                    world.do_step()
            finally:
                reset_framework_logging()
    return states


@pytest.mark.parametrize("encoder", [DenseV3(), DenseV2(), DenseV1(), OneHotE3()])
def test_symmetric_boards_give_the_same_canonical_input(
    engine_states: list[dict[str, Any]], encoder: Encoder
) -> None:
    """On real boards, a rotated or mirrored game produces the identical
    canonical vector -- the property that lets one weight set serve all eight
    orientations.

    Checked on states whose direction features are tie-free *and* whose E3 part
    has a trivial stabiliser. When several symmetries map the features onto the
    same canonical row, ``canonical`` may return either of them, and blocks
    that E3 does not see (distances, tiers) then land in different slots. That
    is the accepted limit of canonicalising on the E3 part alone, recorded in
    dev/dqn.md 5.2: those states share less than full canonicalisation would.
    """
    extractors = {s: Extractor(E3, "best_tier", random.Random(0)) for s in SYMMETRIES}
    compared = 0
    for state in engine_states:
        base = extractors[IDENTITY].extract(Observation.from_game_state(state))
        if any(len(options) > 1 for options in base.options.values()):
            continue
        stabiliser = sum(
            transform_features(base.features, g) == base.features for g in SYMMETRIES
        )
        if stabiliser > 1:
            continue
        index, symmetry = canonical(base.features, E3)
        reference = encoder.encode(E3.decode(index), base, symmetry)
        for other in SYMMETRIES[1:]:
            image = extractors[other].extract(
                Observation.from_game_state(transform_game_state(state, other))
            )
            image_index, image_symmetry = canonical(image.features, E3)
            np.testing.assert_allclose(
                encoder.encode(E3.decode(image_index), image, image_symmetry),
                reference,
                atol=1e-6,
                err_msg=f"{encoder.name} differs under {other}",
            )
            compared += 1
    assert compared >= 400, "the sample must actually exercise the property"


def test_dense_carries_information_the_table_cannot(
    engine_states: list[dict[str, Any]],
) -> None:
    """Two boards with the same E3 row but different distances must differ in
    the dense input -- otherwise there is nothing for D9 to learn from."""
    extractor = Extractor(E3, "best_tier", random.Random(0))
    by_row: dict[int, list[tuple[Any, Symmetry]]] = {}
    for state in engine_states:
        sample = extractor.extract(Observation.from_game_state(state))
        index, symmetry = canonical(sample.features, E3)
        by_row.setdefault(index, []).append((sample, symmetry))
    dense, onehot = DenseV1(), OneHotE3()
    split = 0
    for index, samples in by_row.items():
        vectors = [
            dense.encode(E3.decode(index), sample, symmetry)
            for sample, symmetry in samples
        ]
        one = [
            onehot.encode(E3.decode(index), sample, symmetry)
            for sample, symmetry in samples
        ]
        assert all(np.array_equal(one[0], v) for v in one), "same row, same one-hot"
        split += any(not np.array_equal(vectors[0], v) for v in vectors)
    assert split > 0, "no table row was split by the dense input"


def test_dense_v2_adds_an_approach_direction_in_the_canonical_frame() -> None:
    """The block that keeps the agent from going blind once the board is empty:
    which way the nearest opponent is, written in the canonical action frame,
    and zero when there is no reachable opponent at all."""
    subject = DenseV2()
    assert subject.dim == DenseV1().dim + 4
    assert subject.schema_id != DenseV1().schema_id
    sample = extracted(approach_options=frozenset({0}), opponent_distance=11)
    for symmetry in SYMMETRIES:
        vector = subject.encode(Features(mask=63), sample, symmetry)
        block = DenseV1().dim
        # raw UP is the way; it must land in UP's image and nowhere else.
        expected = [0.0] * 4
        expected[symmetry.direction(0)] = 1.0
        assert list(vector[block : block + 4]) == expected
        # everything dense_v1 already said is unchanged.
        np.testing.assert_array_equal(
            vector[:block], DenseV1().encode(Features(mask=63), sample, symmetry)
        )
    none = subject.encode(Features(), extracted())
    assert not none[DenseV1().dim :].any()


def test_dense_v2_sees_opponents_that_opp_dir_cannot(
    engine_states: list[dict[str, Any]],
) -> None:
    """The point of the block: it points somewhere in states where E3's own
    opponent direction has given up.

    On a board with crates the two can *both* be empty, and honestly so: an
    opponent walled in behind crates is not reachable, and there is no
    direction to give. The block earns its place on a board without crates --
    which is the situation it was added for, after the last crate is gone.
    """
    extractor = Extractor(E3, "best_tier", random.Random(0))
    blind = informed = 0
    for state in engine_states:
        sample = extractor.extract(Observation.from_game_state(state))
        if sample.features.opp_dir != NONE_DIRECTION:
            continue
        blind += 1
        informed += bool(sample.approach_options)
    assert blind > 20, "the sample must contain states with no hunt direction"
    assert informed > 0.25 * blind, f"only {informed}/{blind} got a direction"


def test_dense_v2_points_somewhere_whenever_a_path_exists() -> None:
    """``coin-heaven`` has no crates, so the only thing that can cut the board
    is a live bomb. With none on it the direction must always be there; with
    bombs down an opponent can genuinely be unreachable, and then reporting
    "none" is the truth rather than a gap."""
    extractor = Extractor(E3, "best_tier", random.Random(0))
    open_board = open_blind = bombed = bombed_blind = 0
    with quiet_logging():
        world: Any = create_world(WorldConfig(("bfs_agent",) * 4, "coin-heaven", 21))
        try:
            world.new_round()
            world.user_input = None
            while world.running:
                for agent in world.active_agents:
                    observation = Observation.from_game_state(
                        world.get_state_for_agent(agent)
                    )
                    if not observation.others:
                        continue
                    sample = extractor.extract(observation)
                    blind = not sample.approach_options
                    if observation.bombs:
                        bombed += 1
                        bombed_blind += blind
                    else:
                        open_board += 1
                        open_blind += blind
                        assert sample.opponent_distance is not None
                world.do_step()
        finally:
            reset_framework_logging()
    assert open_board > 100
    assert open_blind == 0, f"{open_blind}/{open_board} open-board states had none"
    # Bombs may cut the board, but they must not do so most of the time.
    if bombed:
        assert bombed_blind < 0.5 * bombed


def test_dense_v3_points_at_the_coin_it_would_win_not_the_nearest() -> None:
    """The block exists because those are different coins a quarter of the
    time; it must carry the winnable one, in the canonical frame."""
    subject = DenseV3()
    assert subject.dim == DenseV2().dim + 4 + len(RACE_SCALARS)
    assert subject.schema_id not in {DenseV2().schema_id, DenseV1().schema_id}
    sample = extracted(
        coin_distance=2,
        approach_options=frozenset({1}),
        own_coin_options=frozenset({2}),
        own_coin_distance=7,
        owned_coins=3,
        contested_coins=4,
        coins_within_5=1,
        coins_within_10=5,
    )
    for symmetry in SYMMETRIES:
        vector = subject.encode(Features(mask=63), sample, symmetry)
        block = DenseV2().dim
        expected = [0.0] * 4
        expected[symmetry.direction(2)] = 1.0
        assert list(vector[block : block + 4]) == expected
        assert vector[block + 4] == pytest.approx(7 / 16)
        assert vector[block + 5] == 0.0
        assert vector[block + 6] == pytest.approx(3 / 9)
        assert vector[block + 7] == pytest.approx(4 / 9)
        # everything dense_v2 already said is unchanged
        np.testing.assert_array_equal(
            vector[:block], DenseV2().encode(Features(mask=63), sample, symmetry)
        )
    none = subject.encode(Features(), extracted())
    block = DenseV2().dim
    assert not none[block : block + 4].any()
    assert (none[block + 4], none[block + 5]) == (1.0, 1.0)


def test_dense_v3_disagrees_with_the_nearest_coin_often_enough_to_matter(
    engine_states: list[dict[str, Any]],
) -> None:
    """The block only earns its dimensions if the coin it points at is
    regularly *not* the one `coin_dir` points at. Measured over real rounds
    that is 23-32% of the steps where a coin is visible; this guards the
    property, not the exact rate."""
    extractor = Extractor(E3, "best_tier", random.Random(0))
    visible = disagreed = contested = 0
    for state in engine_states:
        sample = extractor.extract(Observation.from_game_state(state))
        if sample.coin_distance is None:
            continue
        visible += 1
        contested += bool(sample.contested_coins)
        if sample.own_coin_distance != sample.coin_distance:
            disagreed += 1
        # A coin we win the race for is never nearer than the nearest coin.
        if sample.own_coin_distance is not None:
            assert sample.own_coin_distance >= sample.coin_distance
        assert sample.owned_coins + sample.contested_coins <= sample.coins_visible
    assert visible > 50, "the sample must contain steps with a visible coin"
    assert contested > 0, "no coin was ever contested"
    assert disagreed > 0, "the winnable coin was always the nearest coin"
