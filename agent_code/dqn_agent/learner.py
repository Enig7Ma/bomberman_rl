"""CPU Double DQN, separate from the NumPy-only play path.

The caller owns transition counts, stage lengths, warm-up and train_every.
Only update() increments updates and synchronises target, once per interval.
No replay insertion, game callbacks or checkpoint persistence here.
"""

import copy
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional as F

from .config import Config
from .encoder import Encoder
from .network import QNetwork, masked_greedy
from .replay import ReplayBatch, ReplayBuffer

# Torch's installed stubs leave these NumPy arguments untyped.
from_numpy = cast(Callable[[NDArray[np.generic]], Tensor], cast(Any, torch).from_numpy)


class QNet(nn.Module):
    """32 (or encoder dim) -> 128 -> 128 -> 6, float32 on CPU.

    Linear's default uniform +/-1/sqrt(fan_in), using a private generator.
    Explicit Parameters avoid nn.Linear's implicit global-RNG initialisation.
    """

    def __init__(self, input_dim: int, seed: int | None) -> None:
        super().__init__()
        if type(input_dim) is not int or input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.input_dim = input_dim
        generator = torch.Generator(device="cpu")
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)
        self.generator = generator
        self.W0 = nn.Parameter(
            torch.empty((128, input_dim), dtype=torch.float32, device="cpu")
        )
        self.b0 = nn.Parameter(torch.empty(128, dtype=torch.float32, device="cpu"))
        self.W1 = nn.Parameter(
            torch.empty((128, 128), dtype=torch.float32, device="cpu")
        )
        self.b1 = nn.Parameter(torch.empty(128, dtype=torch.float32, device="cpu"))
        self.W2 = nn.Parameter(torch.empty((6, 128), dtype=torch.float32, device="cpu"))
        self.b2 = nn.Parameter(torch.empty(6, dtype=torch.float32, device="cpu"))
        with torch.no_grad():
            for weight, bias in (
                (self.W0, self.b0),
                (self.W1, self.b1),
                (self.W2, self.b2),
            ):
                bound = 1 / math.sqrt(weight.shape[1])
                weight.uniform_(-bound, bound, generator=generator)
                bias.uniform_(-bound, bound, generator=generator)

    def forward(self, x: Tensor) -> Tensor:
        h = F.relu(F.linear(x, self.W0, self.b0))
        h = F.relu(F.linear(h, self.W1, self.b1))
        return F.linear(h, self.W2, self.b2)

    def values(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        if (
            x.shape != (self.input_dim,)
            or x.dtype != np.float32
            or not np.isfinite(x).all()
        ):
            raise ValueError("expected finite float32 input matching encoder dimension")
        with torch.no_grad():
            return cast(NDArray[np.float32], self.forward(from_numpy(x)).numpy().copy())


def export_numpy(net: QNet) -> dict[str, NDArray[np.float32]]:
    """Independent weights in the inference layout; no optimizer/checkpoint state."""
    return {
        name: cast(NDArray[np.float32], parameter.detach().cpu().numpy().copy())
        for name, parameter in net.named_parameters()
    }


@dataclass(frozen=True)
class UpdateStats:
    loss: float
    mean_abs_td: float
    grad_norm: float  # before clipping
    updates: int
    target_synced: bool


class Learner:
    def __init__(self, encoder: Encoder, config: Config) -> None:
        torch.set_num_threads(1)
        self.encoder = encoder
        self.config = config
        self.online = QNet(encoder.dim, config.init_seed)
        self.target = copy.deepcopy(self.online)
        self.target.requires_grad_(False)
        self.target.eval()
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=config.lr, eps=1e-8
        )
        self._updates = 0
        self.action_rng = random.Random(config.seed)
        self.replay_rng = np.random.default_rng(
            random.Random(config.seed).getrandbits(128)
        )

    @property
    def updates(self) -> int:
        return self._updates

    def epsilon(self, transitions: int, stage_transitions: int) -> float:
        """Stage-relative transition count supplied by caller; no counter mutation."""
        if type(transitions) is not int or transitions < 0:
            raise ValueError("transitions must be nonnegative")
        if type(stage_transitions) is not int or stage_transitions <= 0:
            raise ValueError("stage_transitions must be positive")
        fraction = min(
            1.0, transitions / (stage_transitions * self.config.epsilon_fraction)
        )
        return self.config.epsilon_start + fraction * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def select(
        self,
        x: NDArray[np.float32],
        allowed: Sequence[int],
        *,
        transitions: int,
        stage_transitions: int,
        evaluate: bool = False,
    ) -> int:
        if not allowed or any(type(a) is not int or not 0 <= a < 6 for a in allowed):
            raise ValueError("expected nonempty allowed action indices in 0..5")
        epsilon = 0.0 if evaluate else self.epsilon(transitions, stage_transitions)
        if epsilon > 0 and self.action_rng.random() < epsilon:
            return self.action_rng.choice(allowed)
        return masked_greedy(self.online.values(x), allowed, self.action_rng)

    def sample(self, replay: ReplayBuffer) -> ReplayBatch:
        return replay.sample(
            self.config.batch_size,
            self.replay_rng,
            gamma=self.config.gamma,
            c_coin=self.config.c_coin,
            crate_aid=self.config.crate_aid,
            death_aid=self.config.death_aid,
            bomb_aid=self.config.bomb_aid,
            spot_potential=self.config.spot_potential,
            attack_aid=self.config.attack_aid,
            hunt_potential=self.config.hunt_potential,
        )

    def _validate(self, batch: ReplayBatch) -> None:
        n = len(batch.r)
        if n == 0:
            raise ValueError("empty batch")
        for array, shape, dtype in (
            (batch.x, (n, self.encoder.dim), np.float32),
            (batch.x_next, (n, self.encoder.dim), np.float32),
            (batch.a, (n,), np.uint8),
            (batch.r, (n,), np.float32),
            (batch.done, (n,), np.bool_),
            (batch.mask_next, (n, 6), np.bool_),
            (batch.k, (n,), np.uint8),
        ):
            if array.shape != shape or array.dtype != dtype:
                raise ValueError("batch shape or dtype mismatch")
        if (
            (batch.a >= 6).any()
            or not np.isfinite(batch.x).all()
            or not np.isfinite(batch.r).all()
        ):
            raise ValueError("invalid actions, inputs or rewards")
        live = ~batch.done
        if ((batch.k < 1) | (batch.k > 3)).any():
            raise ValueError("invalid return horizon")
        if not batch.mask_next[live].any(axis=1).all():
            raise ValueError("nonterminal mask must allow an action")
        if not np.isfinite(batch.x_next[live]).all():
            raise ValueError("nonterminal successors must be finite")

    def targets(self, batch: ReplayBatch) -> Tensor:
        """Terminal rows never enter either next-state network or argmax."""
        self._validate(batch)
        with torch.no_grad():
            y = from_numpy(batch.r.copy())
            live = ~batch.done
            if live.any():
                x_next = from_numpy(batch.x_next[live])
                mask = from_numpy(batch.mask_next[live])
                online_q = self.online.forward(x_next).masked_fill(~mask, -torch.inf)
                # Target-selection ties use first allowed index, deterministically.
                actions = online_q.argmax(dim=1, keepdim=True)
                target_q = self.target.forward(x_next).gather(1, actions).squeeze(1)
                discounts = np.power(self.config.gamma, batch.k[live]).astype(
                    np.float32
                )
                y[from_numpy(live)] += from_numpy(discounts) * target_q
            if not torch.isfinite(y).all():
                raise ValueError("nonfinite Double DQN target")
            return y

    def update(self, batch: ReplayBatch) -> UpdateStats:
        y = self.targets(batch)
        q = self.online.forward(from_numpy(batch.x))
        taken = q.gather(1, from_numpy(batch.a.astype(np.int64)).unsqueeze(1)).squeeze(
            1
        )
        loss = F.huber_loss(taken, y, reduction="mean", delta=1.0)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite loss")
        self.optimizer.zero_grad(set_to_none=True)
        cast(Callable[[], None], cast(Any, loss).backward)()
        norm = nn.utils.clip_grad_norm_(
            self.online.parameters(), self.config.grad_clip, error_if_nonfinite=True
        )
        cast(Callable[[], None], cast(Any, self.optimizer).step)()
        self._updates += 1
        synced = self.updates % self.config.target_every == 0
        if synced:
            self.target.load_state_dict(self.online.state_dict())
        return UpdateStats(
            float(loss.detach()),
            float((taken.detach() - y).abs().mean()),
            float(norm),
            self.updates,
            synced,
        )

    def export(self) -> QNetwork:
        """Inference snapshot; the caller supplies stage/transition metadata later."""
        return QNetwork(
            self.encoder,
            export_numpy(self.online),
            {
                "config": asdict(self.config),
                "updates": self.updates,
                "stage": "exported",
            },
        )

    def training_state(self) -> dict[str, Any]:
        return {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "adam": self.optimizer.state_dict(),
            "updates": self.updates,
            "action_rng": self.action_rng.getstate(),
            "replay_rng": self.replay_rng.bit_generator.state,
            "torch_rng": self.online.generator.get_state(),
            "target_torch_rng": self.target.generator.get_state(),
        }

    def restore_training_state(self, state: dict[str, Any]) -> None:
        updates = state["updates"]
        if type(updates) is not int or updates < 0:
            raise ValueError("invalid update counter")
        self.online.load_state_dict(state["online"], strict=True)
        self.target.load_state_dict(state["target"], strict=True)
        if any(
            not torch.isfinite(p).all()
            for net in (self.online, self.target)
            for p in net.parameters()
        ):
            raise ValueError("nonfinite checkpoint weights")
        self.optimizer.load_state_dict(state["adam"])
        self._updates = updates
        self.action_rng.setstate(state["action_rng"])
        self.replay_rng.bit_generator.state = state["replay_rng"]
        self.online.generator.set_state(state["torch_rng"])
        self.target.generator.set_state(
            state.get("target_torch_rng", state["torch_rng"])
        )
