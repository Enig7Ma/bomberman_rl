"""D4 training callbacks, replay sink and round-boundary save hook."""

import time
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

import events as e

from .bookkeeping import Bookkeeper, Transition
from .callbacks import AgentSelf
from .config import REPO_ROOT, metrics_path
from .core.world_model import ACTIONS, Observation
from .features import Extracted
from .learner import Learner, UpdateStats, from_numpy
from .metrics import append_record
from .network import QNetwork, masked_greedy
from .persistence import load_checkpoint, load_replay, save_checkpoint, save_replay
from .probe import Probe, check_q
from .replay import ReplayBuffer, ReplayTransition
from .rewards import Rewards
from .symmetry import Symmetry, from_canonical, to_canonical


class Trainer:
    def __init__(self, agent: AgentSelf) -> None:
        self.agent = agent
        self.config = agent.config
        self.learner = Learner(agent.encoder, self.config)
        self.replay = ReplayBuffer(self.config.replay_size, agent.encoder.dim)
        self.bookkeeper = Bookkeeper[NDArray[np.float32]](self._on_transition)
        self.transitions = self.rounds_trained = self.stage_position = 0
        self.run_id = uuid.uuid4().hex
        self.exact_history = True
        self.resume_mode = "random-init"
        self.parent_checkpoint: str | None = None
        self.last_record: dict[str, Any] = {}
        self.last_save: dict[str, float] = {}
        self.probe: Probe | None = None
        if self.config.probe_path:
            path = Path(self.config.probe_path).expanduser()
            path = path if path.is_absolute() else REPO_ROOT / path
            self.probe = Probe.load(path, agent.encoder.schema_id, agent.encoder.dim)
        checkpoint = agent.model_file.parent / "checkpoint.pt"
        if checkpoint.exists():
            self._restore(checkpoint)
        elif (agent.model_file.parent / "replay.npz").exists():
            raise ValueError(
                "orphan replay without checkpoint; use q_net in a fresh directory"
            )
        elif agent.model_file.exists():
            net = QNetwork.load(agent.model_file, agent.encoder)
            weights = {name: from_numpy(array) for name, array in net.arrays.items()}
            self.learner.online.load_state_dict(weights)
            self.learner.target.load_state_dict(weights)
            self.resume_mode = "numpy-warm-start"
            self.parent_checkpoint = str(agent.model_file)
        self.agent.q_function = self.learner.online
        self._reset_round()
        agent.logger.info(f"DQN training start: {self.resume_mode}")

    def _reset_round(self) -> None:
        self.events: Counter[str] = Counter()
        self.steps = self.forced = self.explored = 0
        self.base = self.shaped_return = 0.0
        self.update_stats: list[UpdateStats] = []
        self.probe_records: list[dict[str, Any]] = []
        self.started = time.perf_counter()
        self.opponents: list[str] = []

    def choose(
        self, x: NDArray[np.float32], allowed: Sequence[int]
    ) -> tuple[int, bool]:
        q = self.learner.online.values(x)
        check_q(q)
        epsilon = self.learner.epsilon(
            self.stage_position, self.config.stage_transitions
        )
        explored = (
            self.config.policy == "random" or self.learner.action_rng.random() < epsilon
        )
        action = (
            self.learner.action_rng.choice(allowed)
            if explored
            else masked_greedy(q, allowed, self.learner.action_rng)
        )
        return action, explored

    def select(
        self, obs: Observation, extracted: Extracted, index: int, symmetry: Symmetry
    ) -> str:
        if obs.round != self.bookkeeper.round:
            self.bookkeeper.begin_round(obs.round)
            self._reset_round()
            self.opponents = [other.name for other in obs.others]
        x = self.agent.encoder.encode(self.agent.encoding.decode(index), extracted)
        allowed = tuple(
            ACTIONS.index(to_canonical(a, symmetry)) for a in extracted.allowed
        )
        self.bookkeeper.observe(
            obs.round, obs.step, x, allowed, extracted.coin_distance
        )
        action, explored = self.choose(x, allowed)
        played = from_canonical(ACTIONS[action], symmetry)
        self.bookkeeper.chose(action, played)
        self.steps += 1
        self.forced += len(allowed) == 1
        self.explored += explored
        return played

    def _on_transition(self, transition: Transition[NDArray[np.float32]]) -> None:
        events = [event for batch in transition.event_batches for event in batch]
        self.events.update(events)
        rewards = Rewards(
            self.config.gamma,
            self.config.c_coin,
            self.config.crate_aid,
            self.config.death_aid,
        )
        unit = Rewards(self.config.gamma, coin_potential=1)
        observed, nxt = transition.observed, transition.next_observed
        base = rewards.base(events)
        phi = unit.potential(observed.coin_distance)
        phi_next = 0.0 if nxt is None else unit.potential(nxt.coin_distance)
        mask = np.zeros(6, dtype=bool)
        if nxt is not None:
            mask[list(nxt.allowed)] = True
        self.replay.push(
            ReplayTransition(
                observed.state,
                transition.action,
                base,
                events.count(e.CRATE_DESTROYED),
                int(e.GOT_KILLED in events or e.KILLED_SELF in events),
                phi,
                phi_next,
                np.zeros_like(observed.state) if nxt is None else nxt.state,
                mask,
                nxt is None,
                self.config.stage,
                self.rounds_trained + 1,
                self.transitions,
            )
        )
        self.base += base
        self.shaped_return += (
            base
            + rewards.aids(events)
            + self.config.c_coin * (self.config.gamma * phi_next - phi)
        )
        self.transitions += 1
        self.stage_position += 1
        if (
            self.transitions >= self.config.warmup
            and self.transitions % self.config.train_every == 0
        ):
            stats = self.learner.update(self.learner.sample(self.replay))
            self.update_stats.append(stats)
            check_q(self.learner.online.values(observed.state))
            if stats.updates % self.config.probe_every == 0:
                record = {"updates": stats.updates, "status": "not-collected-D6"}
                if self.probe is not None:
                    record = {
                        "updates": stats.updates,
                        "status": "measured",
                        **self.probe.measure(self.learner.online),
                    }
                self.probe_records.append(record)

    def finish(self, action: str, events: Sequence[str]) -> None:
        self.bookkeeper.finish(action, events)
        self.rounds_trained += 1
        stats = self.update_stats
        counts = self.events
        self.last_record = {
            "round": self.bookkeeper.round,
            "rounds_trained": self.rounds_trained,
            "stage": self.config.stage,
            "opponents": self.opponents,
            "transitions": self.transitions,
            "updates": self.learner.updates,
            "updates_this_round": len(stats),
            "buffer_fill": len(self.replay),
            "steps": self.steps,
            "epsilon": self.learner.epsilon(
                self.stage_position, self.config.stage_transitions
            ),
            "stage_position": self.stage_position,
            "base_reward": self.base,
            "shaped_return": self.shaped_return,
            "coins": counts[e.COIN_COLLECTED],
            "kills": counts[e.KILLED_OPPONENT],
            "crates": counts[e.CRATE_DESTROYED],
            "bombs": counts[e.BOMB_DROPPED],
            "invalid": counts[e.INVALID_ACTION],
            "died": counts[e.GOT_KILLED] > 0,
            "self_kill": counts[e.KILLED_SELF] > 0,
            "survived": counts[e.SURVIVED_ROUND] > 0,
            "event_counts": dict(counts),
            "forced_fraction": self.forced / self.steps,
            "explored_steps": self.explored,
            "mean_loss": sum(s.loss for s in stats) / len(stats) if stats else None,
            "mean_abs_td": sum(s.mean_abs_td for s in stats) / len(stats)
            if stats
            else None,
            "mean_grad_norm": sum(s.grad_norm for s in stats) / len(stats)
            if stats
            else None,
            "wall_time": time.perf_counter() - self.started,
            "probe": self.probe_records,
            "probe_status": "ready" if self.probe is not None else "not-collected-D6",
            "resume_mode": self.resume_mode,
            "exact_history": self.exact_history,
        }
        if (
            self.rounds_trained % self.config.save_every == 0
            or self.rounds_trained % self.config.replay_save_every == 0
        ):
            self.save(full=self.rounds_trained % self.config.replay_save_every == 0)
            self.last_record["save_seconds"] = dict(self.last_save)
        append_record(metrics_path(), self.last_record)

    def save(self, *, full: bool = True) -> None:
        """Chunk-end hook: full=True saves a consistent pair at a round boundary."""
        if self.bookkeeper.pending is not None:
            raise RuntimeError("save requires a completed round")
        directory = self.agent.model_file.parent
        replay_path = directory / "replay.npz"
        self.last_save = {}
        if full or not replay_path.exists():
            self.last_save["replay"] = save_replay(
                replay_path,
                self.replay,
                run_id=self.run_id,
                transitions=self.transitions,
                schema_id=self.agent.encoder.schema_id,
                exact_history=self.exact_history,
            )
        state = {
            "format_version": 1,
            "schema_id": self.agent.encoder.schema_id,
            "config": asdict(self.config),
            "run_id": self.run_id,
            "transitions": self.transitions,
            "rounds_trained": self.rounds_trained,
            "stage_position": self.stage_position,
            "learner": self.learner.training_state(),
            "feature_rng": self.agent.rng.getstate(),
            "exact_history": self.exact_history,
            "parent_checkpoint": self.parent_checkpoint,
            "probe": None
            if self.probe is None
            else {
                "fingerprint": self.probe.fingerprint,
                "previous": self.probe.previous,
            },
        }
        self.last_save["checkpoint"] = save_checkpoint(
            directory / "checkpoint.pt", state
        )
        net = self.learner.export()
        net.meta.update(
            {
                "transitions": self.transitions,
                "rounds_trained": self.rounds_trained,
                "stage": str(self.config.stage),
                "parent_checkpoint": self.parent_checkpoint,
                "exact_history": self.exact_history,
                "run_id": self.run_id,
            }
        )
        started = time.perf_counter()
        net.save(self.agent.model_file)
        self.last_save["q_net"] = time.perf_counter() - started

    def _restore(self, path: Path) -> None:
        state = load_checkpoint(path)
        if state["schema_id"] != self.agent.encoder.schema_id or state[
            "config"
        ] != asdict(self.config):
            raise ValueError(
                "checkpoint schema/config mismatch; use q_net alone for a new run"
            )
        for key in ("transitions", "rounds_trained", "stage_position"):
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError("invalid checkpoint counters")
        first_update = max(
            1,
            (self.config.warmup + self.config.train_every - 1)
            // self.config.train_every,
        )
        expected_updates = max(
            0, state["transitions"] // self.config.train_every - first_update + 1
        )
        if (
            state["learner"]["updates"] != expected_updates
            or state["stage_position"] != state["transitions"]
        ):
            raise ValueError("checkpoint counters/schedule are inconsistent")
        if state["rounds_trained"] > state["transitions"]:
            raise ValueError("checkpoint round counter exceeds transitions")
        replay, degraded = load_replay(
            path.parent / "replay.npz", state, self.agent.encoder.schema_id
        )
        if (
            replay.capacity != self.config.replay_size
            or replay.input_dim != self.agent.encoder.dim
        ):
            raise ValueError("checkpoint/replay dimensions differ")
        self.learner.restore_training_state(state["learner"])
        self.agent.rng.setstate(state["feature_rng"])
        self.replay = replay
        self.transitions, self.rounds_trained, self.stage_position = (
            state[k] for k in ("transitions", "rounds_trained", "stage_position")
        )
        self.run_id = state["run_id"]
        self.parent_checkpoint = state["parent_checkpoint"]
        self.exact_history = state["exact_history"] and not degraded
        self.resume_mode = (
            "exact-resume" if self.exact_history else "stale-replay-resume"
        )
        if not self.exact_history:
            self.agent.logger.warning(
                "Replay is stale/incomplete: NOT an exact continuation"
            )
        probe = state["probe"]
        if self.probe is not None:
            if probe is None or probe["fingerprint"] != self.probe.fingerprint:
                raise ValueError("checkpoint probe mismatch")
            self.probe.previous = probe["previous"]


def setup_training(self: AgentSelf) -> None:
    self.trainer = Trainer(self)


def game_events_occurred(
    self: AgentSelf,
    old_game_state: Mapping[str, Any],
    self_action: str,
    new_game_state: Mapping[str, Any],
    events: list[str],
) -> None:
    cast(Trainer, self.trainer).bookkeeper.events_occurred(
        int(old_game_state["round"]), int(old_game_state["step"]), self_action, events
    )


def end_of_round(
    self: AgentSelf,
    last_game_state: Mapping[str, Any],
    last_action: str,
    events: list[str],
) -> None:
    cast(Trainer, self.trainer).finish(last_action, events)
