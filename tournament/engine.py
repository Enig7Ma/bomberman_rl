"""Run single tournament rounds directly against the game engine.

The framework's own CLI reuses one world for every round, but ``BombeRLeWorld``
draws from a single RNG for both arena generation *and* the per-step
action-order permutation. Two runs with the same seed therefore agree on round 1
and diverge from round 2 onwards, because round 1 consumed a behaviour-dependent
number of draws. That destroys paired comparison, so this module builds a fresh
world per round with an explicit seed instead.

Doing that exposes a second problem: ``GenericWorld.setup_logging`` and
``AgentRunner.__init__`` *add* a ``FileHandler`` to a module-level logger every
time a world is built, each opened with ``mode="w"``. Left alone, a few thousand
rounds leak that many file descriptors and multiply log I/O, so every round
clears them again.
"""

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import environment
import settings
from environment import BombeRLeWorld, WorldArgs
from tournament.framework import AgentView, agents_of
from tournament.results import AgentRoundResult, RoundResult

REPO_ROOT = Path(environment.__file__).resolve().parent

DEFAULT_LOG_DIR = "logs"

# Loggers the framework attaches file handlers to, beyond the per-agent ones.
_WORLD_LOGGER = "BombeRLeWorld"
_AGENT_LOGGER_SUFFIXES = ("_code", "_wrapper")


@dataclass(frozen=True)
class WorldConfig:
    """Everything that identifies a single round.

    Two configs that compare equal describe the same arena, the same seating and
    the same agents, so this doubles as the unit of scheduling.
    """

    lineup: tuple[str, ...]
    scenario: str
    seed: int
    arm: str = "candidate"
    focus_seat: int = 0


def ensure_repo_cwd() -> None:
    """Change into the repository root.

    ``AgentRunner`` resolves ``agent_code/<name>/logs/`` relative to the process
    working directory and imports agents as ``agent_code.<name>.callbacks``, so
    the harness only works from the repository root.
    """
    if Path.cwd().resolve() != REPO_ROOT:
        os.chdir(REPO_ROOT)


def reset_framework_logging() -> None:
    """Detach and close every file handler the framework added.

    Safe to call when there is nothing to clean up.
    """
    names = [_WORLD_LOGGER]
    names += [
        name
        for name in logging.root.manager.loggerDict
        if name.endswith(_AGENT_LOGGER_SUFFIXES)
    ]
    for name in names:
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()


@contextmanager
def quiet_logging() -> Generator[None]:
    """Raise the framework's log levels for the duration of the block.

    The file handlers are still created and truncated -- that is hardcoded in
    the framework -- but no records reach them, which is where the cost is.
    """
    saved = (settings.LOG_GAME, settings.LOG_AGENT_WRAPPER, settings.LOG_AGENT_CODE)
    silent = logging.CRITICAL + 1
    settings.LOG_GAME = silent
    settings.LOG_AGENT_WRAPPER = silent
    settings.LOG_AGENT_CODE = silent
    try:
        yield
    finally:
        settings.LOG_GAME, settings.LOG_AGENT_WRAPPER, settings.LOG_AGENT_CODE = saved


def _world_args(config: WorldConfig, log_dir: str) -> WorldArgs:
    return WorldArgs(
        no_gui=True,
        fps=15,
        turn_based=False,
        update_interval=0.1,
        save_replay=False,
        replay=None,
        make_video=False,
        continue_without_training=True,
        log_dir=log_dir,
        save_stats=False,
        match_name=None,
        seed=config.seed,
        silence_errors=False,
        scenario=config.scenario,
    )


def create_world(config: WorldConfig, log_dir: str = DEFAULT_LOG_DIR) -> BombeRLeWorld:
    """Build a world for one round. The caller is responsible for cleanup."""
    ensure_repo_cwd()
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    agents = [(code_name, False) for code_name in config.lineup]
    return BombeRLeWorld(_world_args(config, log_dir), agents)


def _agent_result(seat: int, agent: AgentView) -> AgentRoundResult:
    # ``statistics``, ``score`` and ``dead`` are None until the agent's first
    # round starts, so a None here means the round was never played.
    stats = agent.statistics
    if stats is None or agent.score is None or agent.dead is None:
        raise RuntimeError(f"agent {agent.name!r} has no completed round to report")
    return AgentRoundResult(
        code_name=agent.code_name,
        name=agent.name,
        seat=seat,
        score=agent.score,
        coins=stats["coins"],
        kills=stats["kills"],
        suicides=stats["suicides"],
        crates=stats["crates"],
        bombs=stats["bombs"],
        invalid=stats["invalid"],
        moves=stats["moves"],
        steps=stats["steps"],
        survived=not agent.dead,
    )


def collect_result(world: BombeRLeWorld, config: WorldConfig) -> RoundResult:
    """Read the finished round off the world's agent objects."""
    return RoundResult(
        scenario=config.scenario,
        seed=config.seed,
        lineup=config.lineup,
        arm=config.arm,
        focus_seat=config.focus_seat,
        steps=world.step,
        agents=tuple(
            _agent_result(seat, agent) for seat, agent in enumerate(agents_of(world))
        ),
    )


def run_round(config: WorldConfig, log_dir: str = DEFAULT_LOG_DIR) -> RoundResult:
    """Play one round to completion and return its result."""
    world = create_world(config, log_dir)
    try:
        world.new_round()
        while world.running:
            world.do_step()
        return collect_result(world, config)
    finally:
        reset_framework_logging()
