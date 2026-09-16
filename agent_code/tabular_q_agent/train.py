"""Training callbacks, loaded only when the agent runs with ``--train``.

``setup_training`` is called once after ``callbacks.setup``. Per round, the
engine then calls ``game_events_occurred`` after every step the agent survives
and ``end_of_round`` exactly once at the end. Neither has a time limit.

The engine's delivery differs from the rules PDF in ways that matter for
learning (a survivor's last step arrives twice, a death step only in
``end_of_round``, posthumous events later still); ``transitions.Trainer``
handles all of it, and this module only forwards the callbacks, appends the
round's record to ``TABULAR_Q_AGENT_METRICS`` and saves the table.

Run standalone with the stock framework, for example::

    TABULAR_Q_AGENT_MODEL=results/tabular_q/run/q_table.npz \\
    TABULAR_Q_AGENT_METRICS=results/tabular_q/run/metrics.jsonl \\
    python main.py play --no-gui --agents tabular_q_agent --train 1 \\
        --scenario coin-heaven --n-rounds 100

Without ``TABULAR_Q_AGENT_MODEL`` the table in ``model/q_table.npz`` is
trained in place. The stock framework ends a round when the learner dies, so
kills scored after death are lost there; ``training.world.TrainingWorld``
keeps them.
"""

from dataclasses import asdict

from .callbacks import Action, AgentSelf, GameState
from .config import metrics_path
from .metrics import append_record
from .rewards import Rewards
from .transitions import Trainer


def setup_training(self: AgentSelf) -> None:
    """Called once, after ``callbacks.setup``, only in training mode."""
    config = self.config
    rewards = Rewards(
        gamma=config.gamma,
        coin_potential=config.coin_potential,
        crate_aid=config.crate_aid,
        death_aid=config.death_aid,
        bomb_aid=config.bomb_aid,
        spot_potential=config.spot_potential,
    )
    self.trainer = Trainer(
        self.learner, rewards, epsilon=config.epsilon, stage=config.stage
    )


def game_events_occurred(
    self: AgentSelf,
    old_game_state: GameState,
    self_action: Action,
    new_game_state: GameState,
    events: list[str],
) -> None:
    """Called after each step the agent survived, including the last one."""
    trainer = _trainer(self)
    trainer.events_occurred(
        int(old_game_state["round"]), int(old_game_state["step"]), self_action, events
    )


def end_of_round(
    self: AgentSelf,
    last_game_state: GameState,
    last_action: Action,
    events: list[str],
) -> None:
    """Called exactly once per round, whether the agent survived or died."""
    record = _trainer(self).finish(last_action, events)
    append_record(metrics_path(), record)
    if record.rounds_trained % self.config.save_every == 0:
        self.table.meta["config"] = asdict(self.config)
        self.table.save(self.model_file)
    self.logger.info(
        f"round {record.round}: score {record.base_reward:g}, {record.steps} steps, "
        f"{record.visited_states} visited states"
    )


def _trainer(self: AgentSelf) -> Trainer:
    if self.trainer is None:
        raise RuntimeError("training callback before setup_training")
    return self.trainer
