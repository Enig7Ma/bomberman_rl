"""Hand-built result records for tests that must not depend on real games."""

from tournament.results import AgentRoundResult, RoundResult
from tournament.schedule import CANDIDATE_ARM


def make_agent(
    code_name: str, seat: int, score: int, **overrides: object
) -> AgentRoundResult:
    defaults: dict[str, object] = {
        "code_name": code_name,
        "name": code_name,
        "seat": seat,
        "score": score,
        "coins": score,
        "kills": 0,
        "suicides": 0,
        "crates": 0,
        "bombs": 0,
        "invalid": 0,
        "moves": 10,
        "steps": 10,
        "survived": True,
        "latency_mean": 0.001,
        "latency_p99": 0.002,
        "latency_max": 0.003,
        "timeouts": 0,
    }
    defaults.update(overrides)
    return AgentRoundResult(**defaults)  # pyright: ignore[reportArgumentType]


def make_round(
    scores: list[int],
    *,
    seed: int = 0,
    arm: str = CANDIDATE_ARM,
    focus_seat: int = 0,
    focus_name: str = "candidate",
) -> RoundResult:
    names = [
        focus_name if seat == focus_seat else "rival" for seat in range(len(scores))
    ]
    return RoundResult(
        scenario="classic",
        seed=seed,
        lineup=tuple(names),
        arm=arm,
        focus_seat=focus_seat,
        steps=100,
        agents=tuple(
            make_agent(name, seat, score)
            for seat, (name, score) in enumerate(zip(names, scores, strict=True))
        ),
    )
