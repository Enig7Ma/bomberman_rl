"""DQN-only delayed emission, with raw per-step reward components.

No off-policy correction: intermediate actions belong to the behavior policy.
Only completed Bookkeeper transitions enter here; no game observations/events.
"""

import copy
from dataclasses import asdict, replace
from typing import Any

import numpy as np

from .replay import ReplayBuffer, ReplayTransition


class NStep:
    def __init__(self, n_step: int, input_dim: int) -> None:
        if type(n_step) is not int or n_step not in (1, 3):
            raise ValueError("n_step must be 1 or 3")
        self.n_step = n_step
        self.input_dim = input_dim
        self.pending: list[ReplayTransition] = []

    def push(self, transition: ReplayTransition) -> list[ReplayTransition]:
        if transition.reward_steps:
            raise ValueError("accumulator needs original one-step transitions")
        ReplayBuffer(1, self.input_dim).push(transition)  # validate before mutation
        t = copy.deepcopy(transition)
        if self.pending:
            previous = self.pending[-1]
            if (
                t.transition_id != previous.transition_id + 1
                or t.round_id != previous.round_id
                or t.stage != previous.stage
                or not np.array_equal(previous.x_next, t.x)
                or previous.phi_unit_next != t.phi_unit
            ):
                raise ValueError("discontinuous sequence/episode")
        self.pending.append(t)
        ready: list[ReplayTransition] = []
        while self.pending and (t.done or len(self.pending) >= self.n_step):
            sequence = self.pending[: self.n_step]
            first, last = sequence[0], sequence[-1]
            ready.append(
                first
                if self.n_step == 1
                else replace(
                    first,
                    x_next=last.x_next.copy(),
                    mask_next=last.mask_next.copy(),
                    done=last.done,
                    phi_unit_next=0.0 if last.done else last.phi_unit_next,
                    reward_steps=tuple(
                        (
                            s.base,
                            s.crates,
                            s.deaths,
                            s.phi_unit,
                            0.0 if s.done else s.phi_unit_next,
                        )
                        for s in sequence
                    ),
                )
            )
            self.pending.pop(0)
        return ready

    def state(self) -> dict[str, Any]:
        """Only builtin values: safe for torch.load(weights_only=True)."""
        rows: list[dict[str, Any]] = []
        for t in self.pending:
            row = asdict(t)
            for key in ("x", "x_next", "mask_next"):
                row[key] = row[key].tolist()
            rows.append(row)
        return {"n_step": self.n_step, "input_dim": self.input_dim, "pending": rows}

    @classmethod
    def restore(cls, state: dict[str, Any]) -> "NStep":
        result = cls(state["n_step"], state["input_dim"])
        if len(state["pending"]) >= result.n_step:
            raise ValueError("invalid pending queue length")
        for item in state["pending"]:
            row = dict(item)
            for key in ("x", "x_next", "mask_next"):
                row[key] = np.asarray(
                    row[key], dtype=np.bool_ if key == "mask_next" else np.float32
                )
            if row["done"] or result.push(ReplayTransition(**row)):
                raise ValueError("pending queue must contain incomplete sequences")
        return result


def convert_replay(source: ReplayBuffer) -> ReplayBuffer:
    """Strict 1→3 conversion, preserving all starts and physical ring order.

    Refuse gaps, inconsistent successors, or an unfinished trailing episode.
    A ring's oldest retained row needs no predecessor: returns look forwards.
    Never clear, truncate or label a one-step live tail as a three-step return.
    """
    state = source.snapshot()
    rows = state["rows"]
    if (rows["k"] != 1).any():
        raise ValueError("conversion requires one-step replay")
    order = list(range(len(source)))
    if len(source) == source.capacity:
        order = order[state["write"] :] + order[: state["write"]]
    accumulator = NStep(3, source.input_dim)
    converted = ReplayBuffer(source.capacity, source.input_dim)
    previous: ReplayTransition | None = None
    for index in order:
        row = rows[index]
        t = ReplayTransition(
            row["x"].copy(),
            int(row["a"]),
            float(row["base"]),
            int(row["crates"]),
            int(row["deaths"]),
            float(row["phi_unit"]),
            float(row["phi_unit_next"]),
            row["x_next"].copy(),
            row["mask_next"].copy(),
            bool(row["done"]),
            int(row["stage"]),
            int(row["round_id"]),
            int(row["transition_id"]),
        )
        if previous is not None and (
            t.transition_id != previous.transition_id + 1
            or (previous.done and t.round_id <= previous.round_id)
        ):
            raise ValueError("replay gap or invalid episode boundary")
        for ready in accumulator.push(t):
            converted.push(ready)
        previous = t
    if accumulator.pending:
        raise ValueError("replay ends without terminal; cannot convert every start")
    result = converted.snapshot()
    if len(source) == source.capacity:
        result["rows"] = np.roll(result["rows"], state["write"])
        result["write"] = state["write"]
    return ReplayBuffer.from_snapshot(result)
