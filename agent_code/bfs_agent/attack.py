"""Opponent pressure: what a bomb does to the opponents, and how to get close.

The attack reuses the escape search from ``safety`` with the roles swapped.
The opponent model is an assumption, stated here rather than hidden: after our
bomb the opponent may choose any first move, and every other agent gets out of
its way. A bomb only counts as a trap if even that opponent has no escape -- so
a claimed trap is conservative. An opponent that is doomed without our bomb
earns nothing.

Measured against 3 rule_based agents, proven traps are rare; almost all kills
come from *pressure* bombs, whose blast reaches an opponent that could still
escape but often does not. A defensive counterpart -- penalising cells where
an armed opponent could trap us with next step's bomb -- was also tried and
changed nothing measurable, so it was removed.
"""

from .params import Params
from .planning import DistanceCache, Target
from .safety import Board, escape
from .world_model import BOMB_POWER, BOMB_TIMER, Observation, Pos, Timeline, own_bomb


def _manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def opponent_escapes(
    board: Board, timeline: Timeline, pos: Pos, blockers: frozenset[Pos]
) -> bool:
    """Can an agent at ``pos`` outlive ``timeline``, picking its own first move?"""
    blocked = dict.fromkeys(blockers, 0)
    starts = [pos]
    for nxt in board.geometry.neighbors[pos]:
        if nxt not in blockers and board.walkable(nxt):
            starts.append(nxt)
    return any(escape(board, timeline, start, blocked).survives for start in starts)


def bomb_attack_value(
    obs: Observation, board: Board, timeline: Timeline, params: Params
) -> float:
    """What dropping a bomb right here is worth against the opponents."""
    me = obs.me.pos
    armed = timeline.with_bombs(board.geometry, board.crates, [own_bomb(me)])
    blast = board.geometry.blast(me)
    blockers = frozenset({me})
    value = 0.0
    for other in obs.others:
        if _manhattan(other.pos, me) > BOMB_POWER + BOMB_TIMER:
            continue
        if opponent_escapes(board, armed, other.pos, blockers):
            if other.pos in blast:
                value += params.pressure_value
        elif opponent_escapes(board, timeline, other.pos, blockers):
            value += params.trap_value
    return value


def hunt_targets(
    obs: Observation, board: Board, cache: DistanceCache, params: Params
) -> list[Target]:
    """Cells within reach whose blast would cover an opponent where it stands.

    Blast geometry is symmetric -- a cell's blast reaches an opponent exactly
    when the opponent's blast would reach the cell -- so these are the cells
    of each opponent's own blast.

    These only attract movement. They are deliberately not bombing spots: what
    a bomb is worth once we get there is ``bomb_attack_value``, evaluated
    against where the opponent actually is by then. Marking them as spots
    credited ``BOMB`` with ``hunt_value`` on top of that, so the agent bombed
    even when the attack itself was worth nothing.
    """
    if params.hunt_value <= 0:
        return []
    mine = cache.field(obs.me.pos)
    targets: list[Target] = []
    for other in obs.others:
        for cell in board.geometry.blast(other.pos):
            d = mine.get(cell)
            if cell != other.pos and d is not None and d <= params.hunt_radius:
                targets.append(Target(cell, params.hunt_value))
    return targets
