"""Tests for ``tabular_q_agent``'s board symmetries and canonical states."""

import random
from typing import Any

import numpy as np
import pytest

from agent_code.tabular_q_agent.core.world_model import (
    ACTIONS,
    MOVES,
    Observation,
    step,
)
from agent_code.tabular_q_agent.features import (
    DIRECTION_FIELDS,
    ENCODINGS,
    HERE,
    NONE,
    RADIX,
    Encoding,
    Extractor,
    Features,
    mask_actions,
)
from agent_code.tabular_q_agent.symmetry import (
    IDENTITY,
    SYMMETRIES,
    Symmetry,
    canonical,
    compose,
    from_canonical,
    inverse,
    to_canonical,
    transform_features,
    transform_game_state,
    transform_grid,
    transform_mask,
)
from tests.bfs_boards import arena, game_state
from tournament.engine import (
    WorldConfig,
    create_world,
    quiet_logging,
    reset_framework_logging,
)

SIZE = 17
CELLS = [(x, y) for x in range(SIZE) for y in range(SIZE)]
TURN = Symmetry(rotations=1, mirrored=False)
MIRROR = Symmetry(rotations=0, mirrored=True)
E1 = ENCODINGS["E1"]
E3 = ENCODINGS["E3"]


def random_features(encoding: Encoding, rng: random.Random) -> Features:
    values = Features().values()
    for name in encoding.fields:
        values[name] = rng.randrange(RADIX[name])
    return Features(**values)


# --- the group -------------------------------------------------------------


def test_there_are_eight_distinct_symmetries_identity_first() -> None:
    assert SYMMETRIES[0] == IDENTITY
    images = {tuple(s.cell(p, SIZE) for p in CELLS) for s in SYMMETRIES}
    assert len(images) == 8


def test_a_quarter_turn_is_clockwise_in_the_gui_view() -> None:
    turned = [TURN.action(a) for a in ACTIONS]
    assert turned == ["RIGHT", "DOWN", "LEFT", "UP", "WAIT", "BOMB"]
    assert TURN.cell((1, 2), SIZE) == (14, 1)


def test_the_mirror_swaps_left_and_right() -> None:
    mirrored = [MIRROR.action(a) for a in ACTIONS]
    assert mirrored == ["UP", "LEFT", "DOWN", "RIGHT", "WAIT", "BOMB"]
    assert MIRROR.cell((1, 2), SIZE) == (15, 2)


@pytest.mark.parametrize("symmetry", SYMMETRIES)
def test_moves_commute_with_the_cell_map(symmetry: Symmetry) -> None:
    """The direction map is the one the cell map induces."""
    for pos in CELLS:
        for move in MOVES:
            assert step(symmetry.cell(pos, SIZE), symmetry.action(move)) == (
                symmetry.cell(step(pos, move), SIZE)
            )


def test_compose_and_inverse_agree_with_the_cell_maps() -> None:
    for outer in SYMMETRIES:
        assert compose(inverse(outer), outer) == IDENTITY
        assert compose(outer, inverse(outer)) == IDENTITY
        for inner in SYMMETRIES:
            both = compose(outer, inner)
            assert all(
                both.cell(p, SIZE) == outer.cell(inner.cell(p, SIZE), SIZE)
                for p in CELLS
            )


@pytest.mark.parametrize("symmetry", SYMMETRIES)
def test_none_and_here_are_fixed(symmetry: Symmetry) -> None:
    assert symmetry.direction(NONE) == NONE
    assert symmetry.direction(HERE) == HERE


@pytest.mark.parametrize("symmetry", SYMMETRIES)
def test_mask_transform_maps_each_allowed_action(symmetry: Symmetry) -> None:
    images = [transform_mask(bits, symmetry) for bits in range(RADIX["mask"])]
    assert sorted(images) == list(range(RADIX["mask"]))
    for bits, image in enumerate(images):
        expected = {symmetry.action(a) for a in mask_actions(bits)}
        assert set(mask_actions(image)) == expected


# --- boards and game states ------------------------------------------------


@pytest.mark.parametrize("symmetry", SYMMETRIES)
def test_the_stone_walls_are_symmetric(symmetry: Symmetry) -> None:
    assert np.array_equal(transform_grid(arena(), symmetry), arena())


@pytest.mark.parametrize("symmetry", SYMMETRIES)
def test_transform_grid_moves_every_cell(symmetry: Symmetry) -> None:
    grid = np.arange(SIZE * SIZE).reshape(SIZE, SIZE)
    image = transform_grid(grid, symmetry)
    assert all(image[symmetry.cell(p, SIZE)] == grid[p] for p in CELLS)


def test_transform_grid_needs_a_square_board() -> None:
    with pytest.raises(ValueError):
        transform_grid(np.zeros((3, 5)), TURN)


def test_transform_game_state_moves_every_position() -> None:
    state = game_state(
        arena(crates=[(1, 2)]),
        (1, 1),
        others=[(3, 1)],
        bombs=[((5, 1), 2)],
        coins=[(1, 5)],
        explosions=[((7, 1), 1)],
        step=12,
    )
    turned = transform_game_state(state, TURN)

    def cell(pos: tuple[int, int]) -> tuple[int, int]:
        return TURN.cell(pos, SIZE)

    assert turned["self"] == ("me", 0, True, cell((1, 1)))
    assert [other[3] for other in turned["others"]] == [cell((3, 1))]
    assert turned["bombs"] == [(cell((5, 1)), 2)]
    assert turned["coins"] == [cell((1, 5))]
    assert turned["field"][cell((1, 2))] == 1
    assert turned["explosion_map"][cell((7, 1))] == 1
    assert (turned["round"], turned["step"]) == (1, 12)
    assert state["self"][3] == (1, 1)  # the input is left alone


# --- canonical states ------------------------------------------------------


def test_canonical_is_constant_on_an_orbit() -> None:
    rng = random.Random(0)
    for _ in range(500):
        features = random_features(E3, rng)
        index, symmetry = canonical(features, E3)
        assert E3.encode(transform_features(features, symmetry)) == index
        for image in SYMMETRIES:
            assert canonical(transform_features(features, image), E3)[0] == index


def test_canonical_is_idempotent() -> None:
    rng = random.Random(1)
    for _ in range(500):
        index, _ = canonical(random_features(E3, rng), E3)
        assert canonical(E3.decode(index), E3) == (index, IDENTITY)


@pytest.mark.parametrize("symmetry", SYMMETRIES)
def test_actions_round_trip(symmetry: Symmetry) -> None:
    for action in ACTIONS:
        assert from_canonical(to_canonical(action, symmetry), symmetry) == action


def test_images_of_a_state_share_its_canonical_actions() -> None:
    """For a state no symmetry fixes, every image lands on the same table row
    and maps each real action to the same canonical action there."""
    rng = random.Random(2)
    checked = 0
    while checked < 200:
        features = random_features(E3, rng)
        images = [transform_features(features, s) for s in SYMMETRIES]
        if len({E3.encode(image) for image in images}) < len(SYMMETRIES):
            continue
        index, symmetry = canonical(features, E3)
        for image_symmetry, image in zip(SYMMETRIES, images, strict=True):
            image_index, to_row = canonical(image, E3)
            assert image_index == index
            for action in ACTIONS:
                real = image_symmetry.action(action)
                assert to_canonical(real, to_row) == to_canonical(action, symmetry)
        checked += 1


def test_e1_has_the_orbit_count_burnside_predicts() -> None:
    # Per WAIT/BOMB combination (4 of them), count orbits of (move bits, coin
    # direction) by averaging fixed points over the 8 symmetries:
    # identity 16*6, quarter turns 2*2 twice, half turn 4*2, axis mirrors 8*4
    # twice, diagonal mirrors 4*2 twice = 192, so 192 / 8 = 24 orbits each.
    # (Move-bit subsets a symmetry fixes, times the coin values it fixes:
    # NONE and HERE always, plus UP and DOWN under the left-right mirror.)
    classes = {canonical(E1.decode(i), E1)[0] for i in range(E1.n_states)}
    assert len(classes) == 4 * 24


# --- equivariance on real engine states ------------------------------------


@pytest.fixture(scope="module")
def engine_states() -> list[dict[str, Any]]:
    """Observations from headless rounds of four bfs_agents: bombs, crates,
    explosions and opponents in realistic combinations."""
    states: list[dict[str, Any]] = []
    with quiet_logging():
        for scenario, seed in (("classic", 0), ("classic", 1), ("coin-heaven", 2)):
            world: Any = create_world(WorldConfig(("bfs_agent",) * 4, scenario, seed))
            try:
                world.new_round()
                world.user_input = None  # normally set by do_step
                while world.running:
                    if world.step % 10 == 0:
                        for agent in world.active_agents:
                            states.append(world.get_state_for_agent(agent))
                    world.do_step()
            finally:
                reset_framework_logging()
    return states


def test_features_are_equivariant_on_engine_states(
    engine_states: list[dict[str, Any]],
) -> None:
    # One extractor per symmetry, so caches never mix boards of different
    # orientation (they would still be correct, but this keeps the test honest).
    extractors = {s: Extractor(E3, "best_tier", random.Random(0)) for s in SYMMETRIES}
    ties = 0
    directions_seen = dict.fromkeys(DIRECTION_FIELDS, 0)
    for number, state in enumerate(engine_states):
        base = extractors[IDENTITY].extract(Observation.from_game_state(state))
        tie_free = all(len(base.options[name]) <= 1 for name in DIRECTION_FIELDS)
        ties += not tie_free
        for name in DIRECTION_FIELDS:
            directions_seen[name] += bool(base.options[name])
        for symmetry in SYMMETRIES[1:]:
            transformed = transform_game_state(state, symmetry)
            image = extractors[symmetry].extract(
                Observation.from_game_state(transformed)
            )
            expected = transform_features(base.features, symmetry)
            where = f"state {number}, {symmetry}"

            assert image.features.mask == expected.mask, where
            assert image.allowed == tuple(
                a for a in ACTIONS if a in {symmetry.action(b) for b in base.allowed}
            ), where
            assert (
                image.features.bomb_yield,
                image.features.danger,
                image.features.attack,
                image.best_tier,
                image.coin_distance,
            ) == (
                base.features.bomb_yield,
                base.features.danger,
                base.features.attack,
                base.best_tier,
                base.coin_distance,
            ), where
            for name in DIRECTION_FIELDS:
                mapped = {symmetry.direction(d) for d in base.options[name]}
                assert image.options[name] == mapped, f"{where}, {name}"
            if tie_free:
                assert image.features == expected, where
                assert (
                    canonical(image.features, E3)[0]
                    == (canonical(base.features, E3)[0])
                ), where

    # Guard against a vacuous pass: the sample must exercise every feature.
    assert len(engine_states) >= 200
    assert ties > 0
    assert all(count > 0 for count in directions_seen.values()), directions_seen
