"""D3 learning tests; Torch is optional, but installed runs execute all cases."""

import random
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from agent_code.dqn_agent.config import Config
from agent_code.dqn_agent.encoder import OneHotE3
from agent_code.dqn_agent.features import ENCODINGS
from agent_code.dqn_agent.network import QNetwork
from agent_code.dqn_agent.replay import ReplayBatch, ReplayBuffer, ReplayTransition

if TYPE_CHECKING:
    from agent_code.dqn_agent.learner import Learner, QNet

torch = pytest.importorskip("torch")


def make_learner(**changes: object) -> "Learner":
    from agent_code.dqn_agent.learner import Learner

    return Learner(
        OneHotE3(), replace(Config(init_seed=0, seed=0, c_coin=0), **changes)
    )


def dataset(n: int = 64, *, terminal: bool = True) -> ReplayBuffer:
    replay = ReplayBuffer(n, 32)
    rng = np.random.default_rng(17)
    for i in range(n):
        x = rng.normal(size=32).astype(np.float32)
        replay.push(
            ReplayTransition(
                x=x,
                a=i % 6,
                base=float(x[0] - x[1]),
                crates=0,
                deaths=0,
                phi_unit=0,
                phi_unit_next=0,
                x_next=x * 0.5,
                mask_next=np.array([True, True, False, False, False, False]),
                done=terminal,
                stage=0,
                round_id=0,
                transition_id=i,
            )
        )
    return replay


def constant(net: "QNet", values: list[float]) -> None:
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.zero_()
        net.b2.copy_(torch.tensor(values, dtype=torch.float32))


def test_masked_double_target_and_terminals() -> None:
    learner = make_learner(gamma=0.9)
    batch = learner.sample(dataset(3, terminal=False))
    batch.r[:] = 2
    batch.done[::2] = True
    batch.mask_next[::2] = False
    batch.x_next[::2] = np.nan  # must not enter either next-state network
    constant(learner.online, [1, 3, 100, 0, 0, 0])
    constant(learner.target, [10, 4, 999, 0, 0, 0])
    y = learner.targets(batch)
    np.testing.assert_allclose(y.numpy()[1::2], 5.6, atol=1e-6)
    np.testing.assert_array_equal(y.numpy()[::2], batch.r[::2])
    assert not y.requires_grad
    assert all(p.grad is None for p in learner.target.parameters())
    batch.done[:] = True
    batch.mask_next[:] = False
    batch.x_next[:] = np.nan
    np.testing.assert_array_equal(learner.targets(batch).numpy(), batch.r)
    batch.done[0] = False
    with pytest.raises(ValueError, match="nonterminal mask"):
        learner.update(batch)
    assert learner.updates == 0


def test_output_gradients_huber_and_clipping() -> None:
    learner = make_learner(grad_clip=0.01)
    batch = learner.sample(dataset())
    batch.a[:] = 2
    batch.a[::2] = 4
    batch.r[:] = 10
    before = learner.online.forward(torch.tensor(batch.x)).detach().numpy()
    residual = before[np.arange(64), batch.a] - batch.r
    expected_loss = np.where(
        np.abs(residual) <= 1, 0.5 * residual**2, np.abs(residual) - 0.5
    ).mean()
    stats = learner.update(batch)
    assert stats.loss == pytest.approx(float(expected_loss), rel=1e-6)
    gradient = learner.online.W2.grad
    assert gradient is not None
    for a in (0, 1, 3, 5):
        assert torch.count_nonzero(gradient[a]) == 0
    for a in (2, 4):
        assert torch.count_nonzero(gradient[a]) > 0
    total = sum(
        float(p.grad.square().sum())
        for p in learner.online.parameters()
        if p.grad is not None
    )
    assert total**0.5 <= 0.010001 < stats.grad_norm
    assert all(
        p.grad is None and not p.requires_grad for p in learner.target.parameters()
    )


def test_target_sync_and_update_counter() -> None:
    learner = make_learner(target_every=3)
    before = [p.clone() for p in learner.target.parameters()]
    assert all(
        torch.equal(a, b) and a.data_ptr() != b.data_ptr()
        for a, b in zip(
            learner.online.parameters(), learner.target.parameters(), strict=True
        )
    )
    batch = learner.sample(dataset())
    for step in range(1, 7):
        stats = learner.update(batch)
        assert stats.updates == learner.updates == step
        assert stats.target_synced == (step % 3 == 0)
        if stats.target_synced:
            assert all(
                torch.equal(a, b)
                for a, b in zip(
                    learner.online.parameters(),
                    learner.target.parameters(),
                    strict=True,
                )
            )
            before = [p.clone() for p in learner.target.parameters()]
        else:
            assert all(
                torch.equal(a, b)
                for a, b in zip(before, learner.target.parameters(), strict=True)
            )


@pytest.mark.parametrize(
    "count,expected", [(0, 0.3), (300, 0.175), (600, 0.05), (1000, 0.05), (2000, 0.05)]
)
def test_epsilon_schedule(count: int, expected: float) -> None:
    learner = make_learner()
    assert learner.epsilon(count, 1000) == pytest.approx(expected)
    assert learner.updates == 0


def test_private_rngs_selection_and_initialisation() -> None:
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    first, second = make_learner(), make_learner()
    different = make_learner(init_seed=1)
    assert not torch.equal(first.online.W0, different.online.W0)
    assert torch.get_num_threads() == 1
    x = np.zeros(32, dtype=np.float32)
    for learner in (first, second):
        constant(learner.online, [1, 1, 100, 0, 0, 0])

    def draws(learner: "Learner") -> list[int]:
        return [
            learner.select(x, [0, 1], transitions=0, stage_transitions=1000)
            for _ in range(100)
        ]

    assert draws(first) == draws(second)
    assert set(draws(first)) == {0, 1}
    # Replay RNG consumption does not affect the separate action RNG.
    first.sample(dataset())
    for _ in range(100):
        second.select(x, [0, 1], transitions=0, stage_transitions=1000)
    assert draws(first) == draws(second)
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0] and numpy_before[2:] == numpy_after[2:]
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    assert torch.equal(torch_before, torch.get_rng_state())
    constant(first.online, [1, 2, 100, 0, 0, 0])
    assert {
        first.select(x, [0, 1], transitions=0, stage_transitions=1, evaluate=True)
        for _ in range(40)
    } == {1}


def test_overfit_64_fixed_terminal_transitions() -> None:
    learner = make_learner()
    # Replace sampled rows with 64 distinct, fixed synthetic terminal transitions.
    rng = np.random.default_rng(23)
    batch = learner.sample(dataset())
    batch.x[:] = rng.normal(size=(64, 32)).astype(np.float32)
    batch.a[:] = np.arange(64) % 6
    batch.r[:] = rng.uniform(-2, 2, size=64).astype(np.float32)
    batch.base[:] = batch.r
    batch.transition_id[:] = np.arange(64)
    stats = learner.update(batch)
    for _ in range(999):
        stats = learner.update(batch)
    assert stats.loss < 1e-3


def test_corridor_values_and_greedy_policy() -> None:
    learner = make_learner(gamma=0.9, target_every=50)
    replay = ReplayBuffer(9, 32)
    identity = np.eye(32, dtype=np.float32)
    for cell in range(5):
        for action in [1] if cell == 0 else [1, 3]:
            nxt = cell + (1 if action == 1 else -1)
            terminal = nxt == 5
            mask = np.zeros(6, dtype=bool)
            mask[[1] if nxt == 0 else [1, 3]] = True
            replay.push(
                ReplayTransition(
                    identity[cell],
                    action,
                    float(terminal),
                    0,
                    0,
                    0,
                    0,
                    identity[nxt],
                    mask,
                    terminal,
                    0,
                    0,
                    cell * 2 + action,
                )
            )
    for _ in range(2500):
        learner.update(learner.sample(replay))
    for cell in range(5):
        allowed = [1] if cell == 0 else [1, 3]
        assert (
            learner.select(
                identity[cell],
                allowed,
                transitions=0,
                stage_transitions=1,
                evaluate=True,
            )
            == 1
        )
        assert learner.online.values(identity[cell])[1] == pytest.approx(
            0.9 ** (4 - cell), abs=0.05
        )


def test_export_parity_and_independence(tmp_path: Path) -> None:
    learner = make_learner()
    learner.update(learner.sample(dataset()))
    exported = learner.export()
    path = tmp_path / "q_net.npz"
    exported.save(path)
    loaded = QNetwork.load(path, OneHotE3())
    rng = np.random.default_rng(6)
    vectors = [rng.normal(size=32).astype(np.float32) for _ in range(100)]
    vectors += [
        OneHotE3().encode(ENCODINGS["E3"].decode(int(i)))
        for i in rng.integers(ENCODINGS["E3"].n_states, size=100)
    ]
    for x in vectors:
        np.testing.assert_allclose(
            learner.online.values(x), loaded.values(x), atol=1e-5, rtol=1e-5
        )
    assert loaded.meta["updates"] == 1
    saved = loaded.arrays["W0"].copy()
    exported.arrays["W0"].fill(0)
    np.testing.assert_array_equal(learner.export().arrays["W0"], saved)


def test_500_updates_are_bit_exact() -> None:
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    first, second = make_learner(target_every=100), make_learner(target_every=100)
    replay1, replay2 = dataset(64, terminal=False), dataset(64, terminal=False)
    for _ in range(500):
        batch1 = first.sample(replay1)
        batch2 = second.sample(replay2)
        np.testing.assert_array_equal(batch1.transition_id, batch2.transition_id)
        assert first.update(batch1) == second.update(batch2)
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            first.online.parameters(), second.online.parameters(), strict=True
        )
    )
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            first.target.parameters(), second.target.parameters(), strict=True
        )
    )
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0] and numpy_before[2:] == numpy_after[2:]
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    assert torch.equal(torch_before, torch.get_rng_state())


@pytest.mark.parametrize("count,budget", [(-1, 100), (True, 100), (0, 0), (0, -1)])
def test_invalid_schedule(count: int, budget: int) -> None:
    with pytest.raises(ValueError):
        make_learner().epsilon(count, budget)


@pytest.mark.parametrize("fault", ["action", "shape", "dtype", "nan", "successor"])
def test_invalid_batch_does_not_update(fault: str) -> None:
    learner = make_learner()
    batch: ReplayBatch = learner.sample(dataset(terminal=False))
    if fault == "action":
        batch.a[0] = 6
    elif fault == "shape":
        batch = replace(batch, x=batch.x[:, :3])
    elif fault == "dtype":
        batch = replace(batch, done=np.zeros(64, dtype=np.uint8))
    elif fault == "nan":
        batch.r[0] = np.nan
    else:
        batch.x_next[0] = np.nan
    with pytest.raises(ValueError):
        learner.update(batch)
    assert learner.updates == 0
