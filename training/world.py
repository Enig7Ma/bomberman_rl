"""Train-mode worlds with a stop rule that keeps posthumous credit.

With ``--train`` the stock framework ends a round the moment no training agent
is alive. A bomb the learner dropped just before dying can still kill an
opponent -- the tournament scores that kill -- but training never sees it.
Playing the round out with ``--continue-without-training`` fixes that at the
cost of up to ~390 steps of other agents' play per death.

``TrainingWorld`` sits in between: once no training agent is alive, the round
stops as soon as no bomb and no *dangerous* explosion owned by a training
agent remains. Every other stopping condition of the engine is unchanged. The
dead learner's final ``end_of_round`` then carries the posthumous events.
"""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast
from unittest.mock import patch

from environment import BombeRLeWorld, WorldArgs
from tournament.engine import ensure_repo_cwd


class _Agent(Protocol):
    train: bool


class _Bomb(Protocol):
    owner: _Agent


class _Explosion(Protocol):
    owner: _Agent

    def is_dangerous(self) -> bool: ...


class TrainingWorld(BombeRLeWorld):
    def time_to_stop(self) -> bool:
        if bool(super().time_to_stop()):
            return True
        agents = cast(list[_Agent], self.agents)
        active = cast(list[_Agent], self.active_agents)
        if not any(agent.train for agent in agents):
            return False
        if any(agent.train for agent in active):
            return False
        bombs = cast(list[_Bomb], self.bombs)
        explosions = cast(list[_Explosion], self.explosions)
        if any(bomb.owner.train for bomb in bombs):
            return False
        if any(x.owner.train and x.is_dangerous() for x in explosions):
            return False
        self.logger.info(
            "No training agent or training agent's bomb left, wrap up round"
        )
        return True


def create_training_world(
    lineup: Sequence[str],
    scenario: str,
    seed: int,
    *,
    train_seats: int = 1,
    log_dir: str = "logs",
) -> TrainingWorld:
    """A headless world whose first ``train_seats`` agents train.

    ``continue_without_training`` is set so the engine's own training stop
    rule stays out of the way of ``TrainingWorld``'s.
    """
    ensure_repo_cwd()
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    args = WorldArgs(
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
        seed=seed,
        silence_errors=False,
        scenario=scenario,
    )
    agents = [(code_name, seat < train_seats) for seat, code_name in enumerate(lineup)]
    # AgentRunner hardcodes agent_code/<name>/logs and opens files with "w".
    # Route their FileHandlers during construction, before any writes, so spawned
    # runs cannot truncate one another's logs. World logs already use log_dir.
    destination = Path(log_dir).resolve() / "agents"

    class RunFileHandler(logging.FileHandler):
        def __init__(
            self,
            filename: str | Path,
            mode: str = "a",
            encoding: str | None = None,
            delay: bool = False,
            errors: str | None = None,
        ) -> None:
            path = Path(filename)
            if "agent_code" in path.parts:
                path = destination / path.parent.parent.name / path.name
                path.parent.mkdir(parents=True, exist_ok=True)
            super().__init__(path, mode, encoding, delay, errors)

    with patch("logging.FileHandler", RunFileHandler):
        return TrainingWorld(args, agents)


def play_round(world: BombeRLeWorld) -> None:
    world.new_round()
    while world.running:
        world.do_step()
