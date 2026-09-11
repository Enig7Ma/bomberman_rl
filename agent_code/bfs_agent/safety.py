"""Time-expanded escape search and the safe-action filter.

The search runs over ``(cell, offset)``: from each cell the agent may wait, or
step to a neighbour that is enterable at that offset (no wall, no crate that is
still standing, no bomb, no obstacle). A state dies if its cell is lethal at
its offset. An action is *survivable* if some state is still alive at the
timeline's horizon, i.e. after every known hazard has burnt out. Each search
is one breadth-first sweep of at most ``horizon`` layers, so it is cheap enough
to run several times per action.

Other agents are the weak point of any such model, because they move and bomb
between our observations. Measured, they are the *only* weak point: every
self-kill in a 40-round post-mortem of the first version happened because an
opponent took a tile of the planned escape, either by winning the random
action order for that tile or by walking into the corridor a step later. So
the approximations are graded instead of picking one (``Assessment.tier``):

- *optimistic*: other agents block only at offset 0 and then vacate;
- *static*: other agents stay where they were observed;
- *threatened*: static, plus every nearby armed opponent drops a bomb at
  offset 0 on its current cell;
- *contested*: threatened, plus an opponent may occupy any cell it can reach
  in time. A cell an opponent reaches in ``d`` moves cannot be entered from
  offset ``d - 1`` on -- at ``d - 1`` both agents would move in during the same
  step and the random action order decides. Waiting on a cell is always
  allowed: nobody can enter the tile we stand on.

None of these is a guarantee -- opponents can bomb later than offset 0 -- which
is why the agent replans every step.
"""

from collections import deque
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass

from .world_model import (
    BOMB,
    BOMB_POWER,
    WAIT,
    Geometry,
    Observation,
    PendingBomb,
    Pos,
    Timeline,
    crate_cells,
    legal_actions,
    own_bomb,
    step,
)

# Opponents farther than this cannot put us in the blast of a bomb dropped now.
DEFAULT_THREAT_RADIUS = BOMB_POWER + 1

_NEVER = 1 << 30


@dataclass(frozen=True)
class Outcome:
    """Result of one escape search."""

    survives: bool
    # Offsets survived: the horizon if ``survives``, else the offset of death.
    survived: int
    # Cells still reachable alive at the horizon -- a measure of escape slack.
    refuges: int


class Board:
    """The static obstacles of one observation, shared by all searches."""

    def __init__(self, obs: Observation, geometry: Geometry) -> None:
        self.geometry = geometry
        self.crates = crate_cells(obs.field)
        self.me = obs.me.pos
        self.others: frozenset[Pos] = frozenset(other.pos for other in obs.others)
        self.bombs: frozenset[Pos] = frozenset(pos for pos, _ in obs.bombs)

    def walkable(self, pos: Pos) -> bool:
        """Free of crates and bombs right now (walls are not in ``neighbors``)."""
        return pos not in self.crates and pos not in self.bombs


def reach_offsets(
    board: Board, sources: Iterable[Pos], barriers: Collection[Pos] = ()
) -> dict[Pos, int]:
    """For each cell, the first offset at which a source agent could enter it.

    An agent ``d`` moves away enters at offset ``d - 1``; its own cell maps to
    ``-1``. ``barriers`` are cells it cannot pass (our own position).
    """
    reach: dict[Pos, int] = {}
    frontier: deque[Pos] = deque()
    for source in sources:
        if source not in reach:
            reach[source] = -1
            frontier.append(source)
    neighbors = board.geometry.neighbors
    while frontier:
        cell = frontier.popleft()
        entered = reach[cell] + 1
        for nxt in neighbors[cell]:
            if nxt in reach or nxt in barriers or not board.walkable(nxt):
                continue
            reach[nxt] = entered
            frontier.append(nxt)
    return reach


def escape(
    board: Board,
    timeline: Timeline,
    start: Pos,
    blocked_from: Mapping[Pos, int] | None = None,
) -> Outcome:
    """Can an agent standing on ``start`` after action 0 outlive ``timeline``?

    ``blocked_from[c] = t`` forbids *entering* ``c`` at offsets ``>= t``.
    """
    lethal = timeline.lethal
    if lethal and start in lethal[0]:
        return Outcome(survives=False, survived=0, refuges=0)

    blocked = blocked_from or {}
    neighbors = board.geometry.neighbors
    crates = board.crates
    crate_open = timeline.crate_open
    bomb_until = timeline.bomb_until

    layer = {start}
    for offset in range(1, timeline.horizon):
        danger = lethal[offset]
        reached: set[Pos] = set()
        for cell in layer:
            if cell not in danger:
                reached.add(cell)
            for nxt in neighbors[cell]:
                if (
                    nxt in reached
                    or nxt in danger
                    or blocked.get(nxt, _NEVER) <= offset
                    or bomb_until.get(nxt, -1) >= offset
                    or (nxt in crates and crate_open.get(nxt, _NEVER) > offset)
                ):
                    continue
                reached.add(nxt)
        if not reached:
            return Outcome(survives=False, survived=offset, refuges=0)
        layer = reached
    return Outcome(survives=True, survived=timeline.horizon, refuges=len(layer))


@dataclass(frozen=True)
class Assessment:
    """How safe one action is under each opponent model."""

    action: str
    optimistic: Outcome
    static: Outcome
    threatened: Outcome
    contested: Outcome

    @property
    def tier(self) -> int:
        """4 = survives opponents racing for our escape cells and dropping
        bombs, 3 = survives their bombs, 2 = survives them standing still,
        1 = survives only if they get out of the way, 0 = no known escape."""
        if self.contested.survives:
            return 4
        if self.threatened.survives:
            return 3
        if self.static.survives:
            return 2
        if self.optimistic.survives:
            return 1
        return 0


def threat_bombs(
    obs: Observation, radius: int = DEFAULT_THREAT_RADIUS
) -> list[PendingBomb]:
    """Bombs that nearby armed opponents could drop with their next action."""
    x, y = obs.me.pos
    return [
        own_bomb(other.pos)
        for other in obs.others
        if other.can_bomb and abs(other.pos[0] - x) + abs(other.pos[1] - y) <= radius
    ]


def assess_actions(
    obs: Observation,
    board: Board,
    timeline: Timeline,
    threats: Sequence[PendingBomb] = (),
) -> list[Assessment]:
    """Grade every legal action."""
    static = dict.fromkeys(board.others, 0)
    contested = reach_offsets(board, board.others, barriers={board.me})
    threatened_base = timeline.with_bombs(board.geometry, board.crates, threats)
    assessments: list[Assessment] = []
    for action in legal_actions(obs):
        if action == BOMB:
            mine = [own_bomb(obs.me.pos)]
            plain = timeline.with_bombs(board.geometry, board.crates, mine)
            threatened = threatened_base.with_bombs(board.geometry, board.crates, mine)
        else:
            plain, threatened = timeline, threatened_base
        start = step(obs.me.pos, action)
        raced = escape(board, threatened, start, contested)
        if start != obs.me.pos and contested.get(start, _NEVER) <= 0 and raced.survives:
            # An opponent next to ``start`` may move in first this very step;
            # then our move fails and we are still standing where we are.
            stay = escape(board, threatened, obs.me.pos, contested)
            if not stay.survives:
                raced = stay
        assessments.append(
            Assessment(
                action=action,
                optimistic=escape(board, plain, start),
                static=escape(board, plain, start, static),
                threatened=escape(board, threatened, start, static),
                contested=raced,
            )
        )
    return assessments


def safest(assessments: Iterable[Assessment]) -> list[Assessment]:
    """The actions in the best available tier -- never empty.

    If nothing survives under any model, fall back to the actions that live
    longest, preferring ``WAIT`` among them. That fallback is not a claim of
    safety; ``callbacks`` logs it.
    """
    ranked = list(assessments)
    best = max(a.tier for a in ranked)
    if best > 0:
        return [a for a in ranked if a.tier == best]
    longest = max(a.optimistic.survived for a in ranked)
    doomed = [a for a in ranked if a.optimistic.survived == longest]
    waits = [a for a in doomed if a.action == WAIT]
    return waits or doomed
