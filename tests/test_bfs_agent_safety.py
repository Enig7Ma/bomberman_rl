"""Escape puzzles with known answers for ``bfs_agent``'s safety layer.

The corridor boards pin down the timing precisely: a bomb dropped now kills
along its blast after the 4th following move, so a corridor that lets the
agent get 4 cells away is survivable and one that stops at 3 is not. An
off-by-one anywhere in the timeline flips one of those two tests.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from agent_code.bfs_agent.safety import (
    Assessment,
    Board,
    assess_actions,
    reach_offsets,
    safest,
    threat_bombs,
)
from agent_code.bfs_agent.world_model import (
    BOMB,
    BOMB_TIMER,
    DOWN,
    RIGHT,
    WAIT,
    Geometry,
    Observation,
    Pos,
    danger_timeline,
)
from tests.bfs_boards import game_state, parse_board

# The agent at the west end can reach (4, 1) at most: 3 cells from the bomb.
DEAD_END = parse_board(
    [
        "######",
        "#....#",
        "######",
    ]
)
# One cell longer: (5, 1) is 4 cells away, outside the blast.
CORRIDOR = parse_board(
    [
        "#######",
        "#.....#",
        "#######",
    ]
)


def assess(
    field: NDArray[np.int64],
    me: Pos,
    *,
    can_bomb: bool = True,
    others: Sequence[Pos] = (),
    others_can_bomb: bool = True,
    bombs: Sequence[tuple[Pos, int]] = (),
    explosions: Sequence[tuple[Pos, int]] = (),
) -> dict[str, Assessment]:
    obs = Observation.from_game_state(
        game_state(
            field,
            me,
            can_bomb=can_bomb,
            others=others,
            others_can_bomb=others_can_bomb,
            bombs=bombs,
            explosions=explosions,
        )
    )
    geometry = Geometry(obs.field)
    board = Board(obs, geometry)
    timeline = danger_timeline(geometry, board.crates, obs.bombs, obs.explosion_map)
    return {
        a.action: a for a in assess_actions(obs, board, timeline, threat_bombs(obs))
    }


def test_bomb_is_fatal_in_a_short_dead_end() -> None:
    graded = assess(DEAD_END, (1, 1))

    assert graded[BOMB].tier == 0
    # The agent runs to the far end and dies there when the blast arrives.
    assert graded[BOMB].optimistic.survived == BOMB_TIMER
    assert graded[RIGHT].tier == 4


def test_bomb_is_survivable_when_the_corridor_outruns_the_blast() -> None:
    graded = assess(CORRIDOR, (1, 1))

    assert graded[BOMB].tier == 4
    assert graded[BOMB].static.refuges == 1  # only (5, 1) is out of reach


def test_bomb_is_survivable_around_a_corner() -> None:
    corner = parse_board(["#####", "#...#", "###.#", "#####"])
    assert assess(corner, (1, 1))[BOMB].tier == 4


def test_only_waiting_survives() -> None:
    """Moving right walks into fire; moving down walks into a corridor that is
    about to explode. Waiting one step, then going right, is the only escape."""
    field = parse_board(
        [
            "######",
            "#....#",
            "#.####",
            "#.####",
            "######",
        ]
    )
    graded = assess(
        field,
        (1, 1),
        can_bomb=False,
        explosions=[((2, 1), 1)],
        bombs=[((1, 3), 1)],
    )

    assert {a for a, g in graded.items() if g.tier > 0} == {WAIT}
    assert graded[RIGHT].optimistic.survived == 0
    assert graded[DOWN].optimistic.survived == 1


def test_escape_through_a_crate_the_blast_removes() -> None:
    field = parse_board(["#########", "#....c..#", "#########"])

    # Without help the crate at (5, 1) seals the corridor.
    assert assess(field, (1, 1))[BOMB].tier == 0
    # A bomb at (7, 1) about to explode clears it just in time.
    assert assess(field, (1, 1), bombs=[((7, 1), 0)])[BOMB].tier == 4


def test_an_opponent_in_the_escape_corridor_is_graded_down() -> None:
    graded = assess(CORRIDOR, (1, 1), others=[(4, 1)], others_can_bomb=False)

    # Escapes only if the opponent gets out of the way.
    assert graded[BOMB].tier == 1
    assert BOMB not in {a.action for a in safest(graded.values())}


def test_an_armed_opponent_at_the_mouth_of_a_dead_end() -> None:
    armed = assess(DEAD_END, (1, 1), others=[(3, 1)], can_bomb=False)
    unarmed = assess(
        DEAD_END, (1, 1), others=[(3, 1)], can_bomb=False, others_can_bomb=False
    )

    assert {g.tier for g in armed.values()} == {2}
    # Unarmed and unable to get past us, it threatens nothing.
    assert {g.tier for g in unarmed.values()} == {4}


def test_an_opponent_that_can_reach_the_escape_first_is_graded_down() -> None:
    """The escape runs east then south; an opponent waiting in the south
    arm could reach (4, 1) at offset 2, one step before we would."""
    field = parse_board(
        [
            "#######",
            "#.....#",
            "#####.#",
            "#####.#",
            "#######",
        ]
    )
    graded = assess(field, (1, 1), others=[(5, 3)], others_can_bomb=False)

    assert graded[BOMB].tier == 3  # fine if the opponent stays put
    assert graded[RIGHT].tier == 4
    assert BOMB not in {a.action for a in safest(graded.values())}


def test_a_move_an_opponent_can_race_for_must_also_survive_failing() -> None:
    """The only exit is (2, 2), and the opponent below can step into it this
    very step. If it wins the action order, our move fails and we stand in
    the blast -- so the move is not contested-safe."""
    field = parse_board(
        [
            "#######",
            "#.....#",
            "##.####",
            "##.####",
            "#######",
        ]
    )
    bomb = [((1, 1), 1)]
    raced = assess(
        field,
        (2, 1),
        can_bomb=False,
        bombs=bomb,
        others=[(2, 3)],
        others_can_bomb=False,
    )
    alone = assess(field, (2, 1), can_bomb=False, bombs=bomb)

    assert raced[DOWN].tier == 3
    assert alone[DOWN].tier == 4


def test_reach_offsets_counts_moves_and_respects_barriers() -> None:
    obs = Observation.from_game_state(game_state(CORRIDOR, (3, 1), others=[(5, 1)]))
    board = Board(obs, Geometry(obs.field))

    reach = reach_offsets(board, [(5, 1)], barriers={(3, 1)})

    assert reach == {(5, 1): -1, (4, 1): 0}
    assert reach_offsets(board, [(5, 1)])[(1, 1)] == 3


def test_far_opponents_are_not_threats() -> None:
    obs = Observation.from_game_state(
        game_state(CORRIDOR, (1, 1), others=[(5, 1)], others_can_bomb=True)
    )
    assert threat_bombs(obs, radius=3) == []
    assert len(threat_bombs(obs, radius=4)) == 1


def test_fallback_is_never_empty_and_prefers_waiting() -> None:
    # Standing on a bomb that explodes after this action, in a closed cell.
    cell = parse_board(["###", "#.#", "###"])
    graded = assess(cell, (1, 1), can_bomb=False, bombs=[((1, 1), 0)])
    choice = safest(graded.values())

    assert [a.action for a in choice] == [WAIT]
    assert choice[0].tier == 0


def test_fallback_picks_the_longest_survival() -> None:
    # A bomb on the agent in a dead end: every continuation dies, but staying
    # in the blast dies at offset 2 whichever way the agent goes.
    graded = assess(DEAD_END, (2, 1), can_bomb=False, bombs=[((1, 1), 2)])

    assert max(g.tier for g in graded.values()) == 0
    assert {a.optimistic.survived for a in safest(graded.values())} == {2}
