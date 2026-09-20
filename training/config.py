"""Curricula: what a training run plays, loaded from JSON and validated up front.

A curriculum file has exactly the shape of the ``Curriculum`` dataclass, so
the copy a run directory keeps (``dataclasses.asdict``) loads back unchanged::

    {
      "name": "coins-then-crates",
      "params": {"encoding": "E3"},
      "chunk_rounds": 50,
      "eval_every": 250,
      "evaluations": [{"preset": "coin-heaven-solo", "seeds": 10, "seed_start": 500}],
      "stages": [
        {"name": "coins", "scenario": "coin-heaven",
         "lineups": [{"opponents": [], "weight": 1.0}],
         "rounds": 500, "epsilon_start": 0.3, "epsilon_end": 0.05,
         "decay_share": 0.6, "replay_share": 0.0}
      ]
    }

- ``params`` are the agent's fixed ``Config`` fields for the whole run.
  ``epsilon``, ``stage``, ``save_every`` and ``seed`` belong to the driver,
  which sets them per chunk.
- ``lineups`` are sampled per chunk by ``weight``. An opponent is an agent
  directory under ``agent_code/`` or ``"frozen"``: a copy of the learner playing
  its newest snapshot (``training.frozen``). A lineup may name its own
  ``scenario``, which mixes scenarios within one stage.
- ``epsilon`` falls linearly from ``epsilon_start`` to ``epsilon_end`` over the
  first ``decay_share`` of the stage's rounds.
- ``replay_share`` of a stage's chunks replay a random earlier stage's scenario
  and lineup, at the current stage's epsilon.
- Stage rounds and ``eval_every`` must be multiples of ``chunk_rounds``: the
  agent saves its table at the end of every chunk, and snapshots are taken at
  chunk boundaries.

Every mistake is reported before any game is played.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, cast

import settings
from agent_code.dqn_agent.config import Config as DQNConfig
from agent_code.tabular_q_agent.config import Config as AgentConfig
from tournament.engine import REPO_ROOT
from tournament.schedule import PRESETS

LEARNER: Final = "tabular_q_agent"
FROZEN: Final = "frozen"
FROZEN_DIR_PREFIX: Final = "tabular_frozen_"
# Agent parameters the driver sets per chunk; a curriculum may not fix them.
DRIVER_PARAMS: Final = frozenset({"epsilon", "stage", "save_every", "seed"})


DQN_DRIVER_PARAMS = {
    "init_seed",
    "stage_transitions",
    "epsilon_start",
    "epsilon_end",
    "epsilon_fraction",
}
DQN_STAGE_PARAMS = {"lr", "gamma", "c_coin", "crate_aid", "death_aid", "grad_clip"}


class CurriculumError(ValueError):
    """A curriculum file that cannot be run as written."""


@dataclass(frozen=True)
class Lineup:
    opponents: tuple[str, ...]
    weight: float = 1.0
    # Plays this lineup in another scenario than the stage's.
    scenario: str | None = None


@dataclass(frozen=True)
class Stage:
    name: str
    scenario: str
    lineups: tuple[Lineup, ...]
    rounds: int
    epsilon_start: float
    epsilon_end: float
    decay_share: float = 0.6
    replay_share: float = 0.0
    transitions: int | None = None
    params: dict[str, Any] = field(default_factory=dict[str, Any])

    def epsilon_at(self, rounds_done: int) -> float:
        """Epsilon after ``rounds_done`` of this stage's rounds."""
        decay_rounds = self.decay_share * self.rounds
        if rounds_done >= decay_rounds:
            return self.epsilon_end
        progress = rounds_done / decay_rounds
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress


@dataclass(frozen=True)
class Evaluation:
    preset: str
    seeds: int
    seed_start: int = 500


@dataclass(frozen=True)
class Curriculum:
    name: str
    params: dict[str, Any]
    chunk_rounds: int
    eval_every: int
    evaluations: tuple[Evaluation, ...]
    stages: tuple[Stage, ...] = field(default_factory=tuple[Stage, ...])

    agent: str = LEARNER
    eval_every_transitions: int = 100_000

    @property
    def total_rounds(self) -> int:
        return sum(stage.rounds for stage in self.stages)

    def agent_config(self) -> AgentConfig | DQNConfig:
        return (DQNConfig if self.agent == "dqn_agent" else AgentConfig)(**self.params)


# --- parsing ---------------------------------------------------------------


def _object(
    raw: object, where: str, required: set[str], optional: set[str] | None = None
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise CurriculumError(f"{where}: expected an object")
    obj = cast(dict[str, Any], raw)
    missing = required - obj.keys()
    unknown = obj.keys() - required - (optional or set())
    if missing:
        raise CurriculumError(f"{where}: missing {sorted(missing)}")
    if unknown:
        raise CurriculumError(f"{where}: unknown {sorted(unknown)}")
    return obj


def _list(raw: object, where: str) -> list[Any]:
    if not isinstance(raw, list | tuple):
        raise CurriculumError(f"{where}: expected a list")
    return list(cast(list[Any] | tuple[Any, ...], raw))


def _str(raw: object, where: str) -> str:
    if not isinstance(raw, str) or not raw:
        raise CurriculumError(f"{where}: expected a non-empty string")
    return raw


def _int(raw: object, where: str, minimum: int) -> int:
    if not isinstance(raw, int) or isinstance(raw, bool) or raw < minimum:
        raise CurriculumError(f"{where}: expected an integer >= {minimum}")
    return raw


def _share(raw: object, where: str, *, below_one: bool = False) -> float:
    number = isinstance(raw, int | float) and not isinstance(raw, bool)
    if not number or not 0.0 <= cast(float, raw) <= 1.0:
        raise CurriculumError(f"{where}: expected a number in [0, 1]")
    value = float(cast(float, raw))
    if below_one and value >= 1.0:
        raise CurriculumError(f"{where}: must be below 1")
    return value


def _scenario(raw: object, where: str) -> str:
    scenario = _str(raw, where)
    if scenario not in settings.SCENARIOS:
        raise CurriculumError(
            f"{where}: {scenario!r} is not one of {sorted(settings.SCENARIOS)}"
        )
    return scenario


def _opponent(raw: object, where: str) -> str:
    name = _str(raw, where)
    if name == FROZEN:
        return name
    if name in (LEARNER, "dqn_agent"):
        raise CurriculumError(
            f"{where}: the learner cannot be an opponent; use {FROZEN!r}"
        )
    if name.startswith((FROZEN_DIR_PREFIX, "dqn_frozen_")):
        raise CurriculumError(
            f"{where}: {name!r} is managed by the driver; use {FROZEN!r}"
        )
    if not (REPO_ROOT / "agent_code" / name / "callbacks.py").is_file():
        raise CurriculumError(f"{where}: no agent directory agent_code/{name}")
    return name


def _lineup(raw: object, where: str) -> Lineup:
    obj = _object(raw, where, {"opponents"}, {"weight", "scenario"})
    opponents = tuple(
        _opponent(name, f"{where}.opponents[{i}]")
        for i, name in enumerate(_list(obj["opponents"], f"{where}.opponents"))
    )
    if len(opponents) > settings.MAX_AGENTS - 1:
        raise CurriculumError(f"{where}: at most {settings.MAX_AGENTS - 1} opponents")
    weight = obj.get("weight", 1.0)
    if (
        not isinstance(weight, int | float)
        or isinstance(weight, bool)
        or weight <= 0
        or not math.isfinite(weight)
    ):
        raise CurriculumError(f"{where}.weight: expected a number > 0")
    scenario: object = obj.get("scenario")
    if scenario is None:
        return Lineup(opponents, float(weight))
    return Lineup(opponents, float(weight), _scenario(scenario, f"{where}.scenario"))


def _stage(
    raw: object, where: str, chunk_rounds: int, first: bool, dqn: bool = False
) -> Stage:
    obj = _object(
        raw,
        where,
        {"name", "scenario", "lineups", "epsilon_start", "epsilon_end"},
        {"rounds", "decay_share", "replay_share", "transitions", "params"},
    )
    scenario = _scenario(obj["scenario"], f"{where}.scenario")
    lineups = tuple(
        _lineup(item, f"{where}.lineups[{i}]")
        for i, item in enumerate(_list(obj["lineups"], f"{where}.lineups"))
    )
    if not lineups:
        raise CurriculumError(f"{where}.lineups: needs at least one lineup")
    rounds = _int(obj.get("rounds", 0), f"{where}.rounds", 0 if dqn else chunk_rounds)
    transitions = obj.get("transitions")
    if dqn:
        transitions = _int(transitions, f"{where}.transitions", 1)
    elif transitions is not None:
        raise CurriculumError("tabular stages use rounds, not transitions")
    if not dqn and rounds % chunk_rounds:
        raise CurriculumError(f"{where}.rounds: must be a multiple of {chunk_rounds}")
    replay_share = _share(
        obj.get("replay_share", 0.0), f"{where}.replay_share", below_one=True
    )
    if first and replay_share > 0.0:
        raise CurriculumError(
            f"{where}.replay_share: the first stage has nothing to replay"
        )
    return Stage(
        name=_str(obj["name"], f"{where}.name"),
        scenario=scenario,
        lineups=lineups,
        rounds=rounds,
        epsilon_start=_share(obj["epsilon_start"], f"{where}.epsilon_start"),
        epsilon_end=_share(obj["epsilon_end"], f"{where}.epsilon_end"),
        decay_share=_share(obj.get("decay_share", 0.6), f"{where}.decay_share"),
        replay_share=replay_share,
        transitions=transitions,
        params=dict(
            _object(
                obj.get("params", {}),
                f"{where}.params",
                set(),
                set(DQN_STAGE_PARAMS if dqn else AgentConfig.__dataclass_fields__)
                - DRIVER_PARAMS
                - {"encoding"},
            )
        ),
    )


def _evaluation(raw: object, where: str) -> Evaluation:
    obj = _object(raw, where, {"preset", "seeds"}, {"seed_start"})
    preset = _str(obj["preset"], f"{where}.preset")
    if preset not in PRESETS:
        raise CurriculumError(
            f"{where}.preset: {preset!r} is not one of {sorted(PRESETS)}"
        )
    return Evaluation(
        preset=preset,
        seeds=_int(obj["seeds"], f"{where}.seeds", 1),
        seed_start=_int(obj.get("seed_start", 500), f"{where}.seed_start", 0),
    )


def parse_curriculum(raw: object) -> Curriculum:
    obj = _object(
        raw,
        "curriculum",
        {"name", "params", "chunk_rounds", "evaluations", "stages"},
        {"agent", "eval_every", "eval_every_transitions"},
    )
    agent = obj.get("agent", LEARNER)
    if agent not in (LEARNER, "dqn_agent"):
        raise CurriculumError("unknown agent")
    dqn = agent == "dqn_agent"
    raw_params: object = obj["params"]
    if not isinstance(raw_params, dict):
        raise CurriculumError("params: expected an object")
    params = cast(dict[str, Any], raw_params)
    reserved = (
        DRIVER_PARAMS | (DQN_DRIVER_PARAMS if dqn else set[str]())
    ) & params.keys()
    if reserved:
        raise CurriculumError(f"params: {sorted(reserved)} are set by the driver")
    try:
        (DQNConfig if dqn else AgentConfig)(**params)
    except (TypeError, ValueError) as error:
        raise CurriculumError(f"params: {error}") from error

    chunk_rounds = _int(obj["chunk_rounds"], "chunk_rounds", 1)
    eval_every = _int(
        obj.get("eval_every", chunk_rounds if dqn else 0), "eval_every", chunk_rounds
    )
    if eval_every % chunk_rounds:
        raise CurriculumError(f"eval_every: must be a multiple of {chunk_rounds}")
    stage_list = _list(obj["stages"], "stages")
    if not stage_list:
        raise CurriculumError("stages: needs at least one stage")
    stages = tuple(
        _stage(item, f"stages[{i}]", chunk_rounds, first=i == 0, dqn=dqn)
        for i, item in enumerate(stage_list)
    )
    for stage in stages:
        try:
            merged = {**params, **stage.params}
            if dqn:
                merged.update(
                    epsilon_start=stage.epsilon_start,
                    epsilon_end=stage.epsilon_end,
                    epsilon_fraction=stage.decay_share,
                )
                DQNConfig(**merged)
            else:
                AgentConfig(**merged)
        except (TypeError, ValueError) as error:
            raise CurriculumError(f"stage {stage.name}: {error}") from error
    if dqn and len(stages) > 256:
        raise CurriculumError("DQN supports at most 256 stages")
    names = [stage.name for stage in stages]
    if len(set(names)) != len(names):
        raise CurriculumError(f"stages: names must be unique, got {names}")
    return Curriculum(
        name=_str(obj["name"], "name"),
        params=dict(params),
        chunk_rounds=chunk_rounds,
        eval_every=eval_every,
        evaluations=tuple(
            _evaluation(item, f"evaluations[{i}]")
            for i, item in enumerate(_list(obj["evaluations"], "evaluations"))
        ),
        stages=stages,
        agent=agent,
        eval_every_transitions=_int(
            obj.get("eval_every_transitions", 100_000), "eval_every_transitions", 1
        ),
    )


def load_curriculum(path: Path) -> Curriculum:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CurriculumError(f"{path}: not valid JSON ({error})") from error
    return parse_curriculum(raw)
