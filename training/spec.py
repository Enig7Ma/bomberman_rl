"""Agent-specific model operations; importing this module never imports Torch."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from agent_code.dqn_agent.config import Config as DQNConfig
from agent_code.dqn_agent.encoder import ENCODERS, Encoder
from agent_code.dqn_agent.network import QNetwork
from agent_code.tabular_q_agent.features import ENCODINGS
from agent_code.tabular_q_agent.qtable import QTable


def _dqn_encoder(params: dict[str, Any]) -> Encoder:
    """The input a curriculum asks for, or the agent's own default."""
    return ENCODERS[str(params.get("encoder", DQNConfig().encoder))]


@dataclass(frozen=True)
class AgentSpec:
    name: str
    prefix: str
    model_file: str
    frozen_prefix: str

    def model_info(self, path: Path, params: dict[str, Any]) -> tuple[int, int]:
        """Training rounds and visited table states (zero for a network)."""
        if self.name == "dqn_agent":
            net = QNetwork.load(path, _dqn_encoder(params))
            return int(net.meta.get("rounds_trained", 0)), 0
        table = QTable.load(path, ENCODINGS[params.get("encoding", "E3")])
        return int(table.meta.get("rounds_trained", 0)), table.visited_states

    def initial_model(self, path: Path, params: dict[str, Any], seed: int) -> None:
        if self.name == "dqn_agent":
            QNetwork.random(_dqn_encoder(params), seed).save(path)
        else:
            QTable.zeros(ENCODINGS[params.get("encoding", "E3")]).save(path)

    def save_chunk(self, agent: object) -> None:
        """Sequential-backend hook, after end_of_round has flushed bookkeeping."""
        handle = cast(Any, agent)
        if self.name == "dqn_agent":
            handle.trainer.save(full=True)
        else:
            handle.table.save(handle.model_file)


SPECS = {
    "tabular_q_agent": AgentSpec(
        "tabular_q_agent", "TABULAR_Q_AGENT", "q_table.npz", "tabular_frozen_"
    ),
    "dqn_agent": AgentSpec("dqn_agent", "DQN_AGENT", "q_net.npz", "dqn_frozen_"),
}
