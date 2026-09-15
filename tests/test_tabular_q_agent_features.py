"""Tests for ``tabular_q_agent``'s feature extraction and state encodings."""

import random
from collections.abc import Sequence
from typing import get_args

import numpy as np
import pytest
from numpy.typing import NDArray

from agent_code.tabular_q_agent.config import ENV_VAR, Config, EncodingName
from agent_code.tabular_q_agent.core.safety import (
    Board,
    assess_actions,
    safest,
    threat_bombs,
)
from agent_code.tabular_q_agent.core.world_model import (
    Geometry,
    Observation,
    Pos,
    danger_timeline,
)
from agent_code.tabular_q_agent.features import (
    DOWN,
    ENCODINGS,
    HERE,
    LEFT,
    NO_ATTACK,
    NONE,
    PRESSURE,
    RADIX,
    RIGHT,
    TRAP,
    Encoding,
    Extracted,
    Extractor,
    Features,
    mask_actions,
    mask_bits,
)
from tests.bfs_boards import arena, game_state, parse_board

CORRIDOR = parse_board(["#######", "#.....#", "#######"])
LONG_CORRIDOR = parse_board(["#########", "#.......#", "#########"])
SPLIT = parse_board(["#######", "#..#..#", "#######"])
COLUMN = parse_board(["###", "#.#", "#.#", "#.#", "###"])
DEAD_END = parse_board(["######", "#....#", "######"])
# A corridor x = 1..5 with a side exit south at x = 3 (as in the bfs_agent tests).
POCKET = parse_board(["#######", "#.....#", "###.###", "#######"])


def extract(
    field: NDArray[np.int64],
    me: Pos,
    *,
    encoding: str = "E3",
    seed: int = 0,
    others: Sequence[Pos] = (),
    others_can_bomb: bool = True,
    bombs: Sequence[tuple[Pos, int]] = (),
    coins: Sequence[Pos] = (),
    explosions: Sequence[tuple[Pos, int]] = (),
) -> Extracted:
    state = game_state(
        field,
        me,
        others=others,
        others_can_bomb=others_can_bomb,
        bombs=bombs,
        coins=coins,
        explosions=explosions,
    )
    extractor = Extractor(ENCODINGS[encoding], "best_tier", random.Random(seed))
    return extractor.extract(Observation.from_game_state(state))


# --- coins -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "me", "coin", "direction", "distance"),
    [
        (CORRIDOR, (1, 1), (4, 1), RIGHT, 3),
        (CORRIDOR, (5, 1), (2, 1), LEFT, 3),
        (COLUMN, (1, 1), (1, 3), DOWN, 2),
    ],
)
def test_coin_dir_points_along_the_corridor(
    field: NDArray[np.int64], me: Pos, coin: Pos, direction: int, distance: int
) -> None:
    extracted = extract(field, me, coins=[coin])
    assert extracted.features.coin_dir == direction
    assert extracted.coin_distance == distance


def test_no_coin_or_an_unreachable_coin_is_none() -> None:
    assert extract(CORRIDOR, (1, 1)).features.coin_dir == NONE
    unreachable = extract(SPLIT, (1, 1), coins=[(4, 1)])
    assert unreachable.features.coin_dir == NONE
    assert unreachable.coin_distance is None


def test_coin_dir_prefers_the_nearest_coin() -> None:
    # (1, 3) is 2 steps south, (5, 1) is 4 steps east.
    assert extract(arena(), (1, 1), coins=[(1, 3), (5, 1)]).features.coin_dir == DOWN


def test_equally_short_first_steps_are_broken_at_random() -> None:
    # (5, 5) is 4 steps from (3, 3) both via (4, 3) and via (3, 4); (4, 4) is a wall.
    seen = {
        extract(arena(), (3, 3), coins=[(5, 5)], seed=seed).features.coin_dir
        for seed in range(40)
    }
    assert seen == {RIGHT, DOWN}


# --- crates ----------------------------------------------------------------


def test_crate_dir_is_here_on_the_best_spot() -> None:
    features = extract(arena(crates=[(1, 2)]), (1, 1)).features
    assert features.crate_dir == HERE
    assert features.bomb_yield == 1


def test_crate_dir_points_to_the_nearest_good_spot() -> None:
    # (1, 1) hits the crate 2 steps west; (1, 3) also does, but 4 steps away.
    features = extract(arena(crates=[(1, 2)]), (3, 1)).features
    assert features.crate_dir == LEFT
    assert features.bomb_yield == 0


def test_no_crates_means_no_crate_dir() -> None:
    assert extract(arena(), (1, 1)).features.crate_dir == NONE


def test_crates_a_live_bomb_will_destroy_do_not_count() -> None:
    field = arena(crates=[(3, 1), (5, 1)])
    assert extract(field, (2, 1)).features.bomb_yield == 2
    # The bomb at (5, 3) will take out (5, 1).
    assert extract(field, (2, 1), bombs=[((5, 3), 3)]).features.bomb_yield == 1


def test_bomb_yield_is_capped() -> None:
    field = arena(crates=[(2, 1), (3, 1), (4, 1), (1, 2), (1, 3)])
    assert extract(field, (1, 1)).features.bomb_yield == 3


# --- danger ----------------------------------------------------------------


@pytest.mark.parametrize("timer", [0, 1, 2, 3, 4])
def test_a_ticking_bomb_in_range_is_danger(timer: int) -> None:
    assert extract(CORRIDOR, (1, 1), bombs=[((3, 1), timer)]).features.danger == 1


def test_a_bomb_out_of_range_is_not_danger() -> None:
    # The blast of (5, 1) reaches x = 2 but not x = 1.
    assert extract(LONG_CORRIDOR, (1, 1), bombs=[((5, 1), 2)]).features.danger == 0


def test_a_live_explosion_on_us_is_danger() -> None:
    features = extract(CORRIDOR, (1, 1), explosions=[((1, 1), 1)]).features
    assert features.danger == 1


# --- opponents -------------------------------------------------------------


def test_a_bomb_at_the_mouth_of_a_dead_end_is_a_trap() -> None:
    features = extract(POCKET, (2, 1), others=[(1, 1)], others_can_bomb=False).features
    assert features.attack == TRAP


def test_an_opponent_that_can_run_is_under_pressure() -> None:
    # From (4, 1) the opponent reaches the side exit (3, 2) in two moves.
    features = extract(POCKET, (2, 1), others=[(4, 1)], others_can_bomb=False).features
    assert features.attack == PRESSURE


def test_no_attack_on_a_doomed_or_distant_opponent() -> None:
    doomed = extract(
        POCKET, (2, 1), others=[(1, 1)], others_can_bomb=False, bombs=[((1, 1), 0)]
    )
    far = extract(arena(), (1, 1), others=[(15, 15)])
    assert doomed.features.attack == NO_ATTACK
    assert far.features.attack == NO_ATTACK


def test_opp_dir_points_to_a_cell_that_threatens_the_opponent() -> None:
    # The blast of (1, 9) reaches (1, 6), 5 steps south of us.
    assert extract(arena(), (1, 1), others=[(1, 9)]).features.opp_dir == DOWN


def test_opp_dir_is_here_when_already_in_position() -> None:
    assert extract(arena(), (1, 1), others=[(1, 3)]).features.opp_dir == HERE


def test_opp_dir_ignores_opponents_beyond_the_hunt_radius() -> None:
    assert extract(arena(), (1, 1), others=[(15, 15)]).features.opp_dir == NONE


# --- mask ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "me", "others"),
    [
        (DEAD_END, (1, 1), []),
        (arena(), (7, 7), []),
        (POCKET, (2, 1), [(4, 1)]),
    ],
)
def test_the_mask_is_the_safest_tier(
    field: NDArray[np.int64], me: Pos, others: list[Pos]
) -> None:
    extracted = extract(field, me, others=others)
    obs = Observation.from_game_state(game_state(field, me, others=others))
    board = Board(obs, Geometry(obs.field))
    timeline = danger_timeline(
        board.geometry, board.crates, obs.bombs, obs.explosion_map
    )
    graded = assess_actions(obs, board, timeline, threat_bombs(obs))

    assert set(extracted.allowed) == {a.action for a in safest(graded)}
    assert mask_actions(extracted.features.mask) == extracted.allowed
    assert extracted.best_tier == max(a.tier for a in graded)


def test_bomb_is_masked_out_in_a_dead_end() -> None:
    assert "BOMB" not in extract(DEAD_END, (1, 1)).allowed


def test_mask_bits_round_trip() -> None:
    for bits in range(RADIX["mask"]):
        assert mask_bits(mask_actions(bits)) == bits


# --- encodings -------------------------------------------------------------


def test_smaller_encodings_leave_their_absent_fields_at_defaults() -> None:
    features = extract(
        arena(crates=[(1, 2)]), (1, 1), others=[(1, 3)], encoding="E1"
    ).features
    assert features == Features(mask=features.mask, coin_dir=features.coin_dir)


def test_state_counts() -> None:
    assert ENCODINGS["E1"].n_states == 64 * 5
    assert ENCODINGS["E2"].n_states == 64 * 5 * 6 * 4 * 2
    assert ENCODINGS["E3"].n_states == 64 * 5 * 6 * 4 * 2 * 6 * 3


def test_every_config_encoding_exists() -> None:
    assert set(ENCODINGS) == set(get_args(EncodingName))


def random_features(encoding: Encoding, rng: random.Random) -> Features:
    values = Features().values()
    for name in encoding.fields:
        values[name] = rng.randrange(RADIX[name])
    return Features(**values)


@pytest.mark.parametrize("name", sorted(ENCODINGS))
def test_encode_and_decode_round_trip(name: str) -> None:
    encoding = ENCODINGS[name]
    rng = random.Random(0)
    for _ in range(2000):
        features = random_features(encoding, rng)
        index = encoding.encode(features)
        assert 0 <= index < encoding.n_states
        assert encoding.decode(index) == features


@pytest.mark.parametrize("name", sorted(ENCODINGS))
def test_encoding_is_a_bijection_at_the_ends(name: str) -> None:
    encoding = ENCODINGS[name]
    top = Features(
        **{**Features().values(), **{f: RADIX[f] - 1 for f in encoding.fields}}
    )
    bottom = Features(**{**Features().values(), **dict.fromkeys(encoding.fields, 0)})
    assert encoding.encode(bottom) == 0
    assert encoding.encode(top) == encoding.n_states - 1


def test_encode_rejects_out_of_range_and_absent_fields() -> None:
    with pytest.raises(ValueError):
        ENCODINGS["E3"].encode(Features(coin_dir=5))
    with pytest.raises(ValueError):
        ENCODINGS["E1"].encode(Features(crate_dir=HERE))
    with pytest.raises(ValueError):
        ENCODINGS["E1"].decode(ENCODINGS["E1"].n_states)


def test_schema_ids_are_stable_and_distinct() -> None:
    ids = [encoding.schema_id for encoding in ENCODINGS.values()]
    assert len(set(ids)) == len(ids)
    assert ids == [encoding.schema_id for encoding in ENCODINGS.values()]
    assert all(len(schema) == 16 for schema in ids)


# --- config ----------------------------------------------------------------


def test_encoding_defaults_to_e3_and_can_be_overridden() -> None:
    assert Config().encoding == "E3"
    assert Config.from_env({ENV_VAR: '{"encoding": "E1"}'}).encoding == "E1"
    with pytest.raises(ValueError):
        Config.from_env({ENV_VAR: '{"encoding": "E4"}'})
