"""Where to go next, among the actions the safety layer allows.

Every candidate action is scored by the targets it brings closer: a target of
value ``v`` that is ``d`` steps away after the action contributes
``v * discount ** d``, and an action is worth its best target. ``WAIT`` and
``BOMB`` leave the agent in place, so a move towards a target beats waiting.

Coins are visited in the order of a planned route rather than nearest-first:
the first coin on the route gets full value and the others ``off_route`` times
that. Measured on ``coin-heaven``, nearest-first is no faster than the
incumbent, and adding a share of *all* targets to the value (to lean towards
clusters) creates local maxima where every move loses value and the agent
stands still.

Distances are shortest walks over cells that are free now. Crates and bombs
block; other agents do not, since they will have moved by the time we arrive
(the first step is already filtered by legality and safety).
"""

import random
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from .params import Params
from .safety import Board
from .world_model import Observation, Pos, step

# Values closer than this are ties, broken at random.
_TIE = 1e-9
# Stand-in distance between cells that cannot reach each other.
UNREACHABLE = 10_000
# Improvement passes over a route; each is O(n^2) in the number of stops.
_ROUTE_PASSES = 4


@dataclass(frozen=True)
class Target:
    pos: Pos
    value: float


def distances(board: Board, starts: Sequence[Pos]) -> dict[Pos, int]:
    """Walking distance from the nearest of ``starts`` to every reachable cell."""
    dist: dict[Pos, int] = {}
    frontier: deque[Pos] = deque()
    for start in starts:
        if start not in dist:
            dist[start] = 0
            frontier.append(start)
    neighbors = board.geometry.neighbors
    while frontier:
        cell = frontier.popleft()
        nxt_dist = dist[cell] + 1
        for nxt in neighbors[cell]:
            if nxt not in dist and board.walkable(nxt):
                dist[nxt] = nxt_dist
                frontier.append(nxt)
    return dist


class DistanceCache:
    """Single-source distance fields, reused until a crate or bomb changes.

    Agents are not obstacles in these fields, so on ``coin-heaven`` (no
    crates, no bombs) every field is computed once per round.
    """

    def __init__(self) -> None:
        self._board: Board | None = None
        self._crates: frozenset[Pos] = frozenset()
        self._bombs: frozenset[Pos] = frozenset()
        self._fields: dict[Pos, dict[Pos, int]] = {}

    def sync(self, board: Board) -> None:
        """Point the cache at this step's board, dropping stale fields."""
        if board.crates != self._crates or board.bombs != self._bombs:
            self._crates = board.crates
            self._bombs = board.bombs
            self._fields.clear()
        self._board = board

    def field(self, source: Pos) -> dict[Pos, int]:
        if self._board is None:
            raise RuntimeError("DistanceCache.sync() must be called first")
        found = self._fields.get(source)
        if found is None:
            found = self._fields[source] = distances(self._board, [source])
        return found

    def between(self, a: Pos, b: Pos) -> int:
        return self.field(a).get(b, UNREACHABLE)


def route_length(start: Pos, route: Sequence[Pos], cache: DistanceCache) -> int:
    total = 0
    here = start
    for stop in route:
        total += cache.between(here, stop)
        here = stop
    return total


def plan_route(start: Pos, stops: Sequence[Pos], cache: DistanceCache) -> list[Pos]:
    """A short open path from ``start`` through every stop.

    Nearest-neighbour construction, then 2-opt (reverse a segment) and or-opt
    (move one stop elsewhere) until neither helps. Distances are symmetric on
    this grid, which both moves rely on.
    """
    remaining = sorted(stops)
    route: list[Pos] = []
    here = start
    while remaining:
        nearest = min(remaining, key=lambda stop: cache.between(here, stop))
        remaining.remove(nearest)
        route.append(nearest)
        here = nearest

    d = cache.between
    for _ in range(_ROUTE_PASSES):
        improved = False
        # 2-opt on the path [start] + route; ``path[i]`` is ``route[i - 1]``.
        path = [start, *route]
        n = len(path) - 1
        for i in range(1, n):
            for j in range(i + 1, n + 1):
                old = d(path[i - 1], path[i])
                new = d(path[i - 1], path[j])
                if j < n:
                    old += d(path[j], path[j + 1])
                    new += d(path[i], path[j + 1])
                if new < old:
                    path[i : j + 1] = path[i : j + 1][::-1]
                    improved = True
        # or-opt: take one stop out and put it where it costs least.
        for k in range(1, n + 1):
            stop = path[k]
            gain = d(path[k - 1], stop)
            if k < n:
                gain += d(stop, path[k + 1]) - d(path[k - 1], path[k + 1])
            rest = path[:k] + path[k + 1 :]
            best_cost, best_at = gain, -1
            for m in range(len(rest)):
                cost = d(rest[m], stop)
                if m + 1 < len(rest):
                    cost += d(stop, rest[m + 1]) - d(rest[m], rest[m + 1])
                if cost < best_cost:
                    best_cost, best_at = cost, m
            if best_at >= 0:
                path = rest[: best_at + 1] + [stop] + rest[best_at + 1 :]
                improved = True
        route = path[1:]
        if not improved:
            break
    return route


class CoinRoute:
    """The coin we are heading for, kept until it is gone or a new one appears.

    Replanning from scratch every step oscillates: two neighbouring cells can
    each make a different route look shortest, and the agent steps back and
    forth between them for the rest of the round (4 of 40 ``coin-heaven``
    rounds before this was added).
    """

    def __init__(self) -> None:
        self.target: Pos | None = None
        self._planned_with: frozenset[Pos] = frozenset()

    def next_coin(self, me: Pos, reachable: Sequence[Pos], cache: DistanceCache) -> Pos:
        coins = frozenset(reachable)
        target = self.target
        if target is None or target not in coins or not coins <= self._planned_with:
            target = plan_route(me, reachable, cache)[0]
            self.target = target
            self._planned_with = coins
        return target


def coin_targets(
    obs: Observation,
    board: Board,
    cache: DistanceCache,
    params: Params,
    route: CoinRoute,
) -> list[Target]:
    """Visible coins we can walk to: the next one on our route at full value.

    A coin some rival reaches strictly before us is discounted.
    """
    mine = cache.field(obs.me.pos)
    reachable = [coin for coin in obs.coins if coin in mine]
    if not reachable:
        return []
    rivals = distances(board, [other.pos for other in obs.others])
    first = route.next_coin(obs.me.pos, reachable, cache)
    targets: list[Target] = []
    for coin in reachable:
        value = params.coin_value * (1.0 if coin == first else params.off_route)
        theirs = rivals.get(coin)
        if theirs is not None and theirs < mine[coin]:
            value *= params.contested_coin
        targets.append(Target(coin, value))
    return targets


def action_values(
    obs: Observation,
    actions: Sequence[str],
    targets: Sequence[Target],
    cache: DistanceCache,
    params: Params,
) -> dict[str, float]:
    """Discounted value of the best target from each action's resulting cell."""
    values: dict[str, float] = {}
    for action in actions:
        dist = cache.field(step(obs.me.pos, action))
        best = 0.0
        for target in targets:
            d = dist.get(target.pos)
            if d is not None:
                best = max(best, target.value * params.discount**d)
        values[action] = best
    return values


def choose(values: dict[str, float], rng: random.Random) -> str:
    """The highest-valued action, ties broken by ``rng``."""
    top = max(values.values())
    return rng.choice(
        [action for action, value in values.items() if value >= top - _TIE]
    )
