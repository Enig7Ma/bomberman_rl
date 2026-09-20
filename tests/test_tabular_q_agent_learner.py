"""Tests for ``tabular_q_agent``'s Q-table storage and learning rule."""

import json
import random
from pathlib import Path

import numpy as np
import pytest

from agent_code.tabular_q_agent import qtable
from agent_code.tabular_q_agent.config import ENV_VAR, Config
from agent_code.tabular_q_agent.core.world_model import ACTIONS
from agent_code.tabular_q_agent.features import ENCODINGS
from agent_code.tabular_q_agent.learner import Learner
from agent_code.tabular_q_agent.qtable import (
    ACTION_COLUMN,
    QTable,
    SchemaMismatchError,
)

E1 = ENCODINGS["E1"]
E2 = ENCODINGS["E2"]
UP, RIGHT, DOWN, LEFT, WAIT, BOMB = range(6)


def make_learner(*, seed: int = 0, gamma: float = 0.9) -> Learner:
    return Learner(
        QTable.zeros(E1),
        gamma=gamma,
        alpha_omega=0.7,
        alpha_min=0.05,
        rng=random.Random(seed),
    )


# --- the update rule -------------------------------------------------------


def test_a_fresh_table_is_all_zeros() -> None:
    table = QTable.zeros(E1)
    assert table.q.shape == table.n.shape == (E1.n_states, len(ACTIONS))
    assert table.q.dtype == np.float32
    assert table.n.dtype == np.uint32
    assert not table.q.any()
    assert table.visited_states == 0


def test_columns_follow_the_action_order() -> None:
    assert [ACTION_COLUMN[a] for a in ACTIONS] == list(range(len(ACTIONS)))


def test_the_first_update_sets_the_value_to_its_target() -> None:
    learner = make_learner()
    delta = learner.update(3, RIGHT, 1.0, None)
    assert delta == 1.0
    assert learner.table.q[3, RIGHT] == 1.0
    assert learner.table.n[3, RIGHT] == 1
    assert learner.table.visited_states == 1


def test_an_update_matches_the_closed_form() -> None:
    learner = make_learner(gamma=0.9)
    table = learner.table
    table.q[0, RIGHT] = 0.5
    table.n[0, RIGHT] = 3
    # RIGHT (9.0) is the largest next value but not allowed there.
    table.q[5] = [2.0, 9.0, 1.0, 0.0, 0.0, 0.0]

    delta = learner.update(0, RIGHT, 0.25, 5, [UP, DOWN])

    target = 0.25 + 0.9 * 2.0
    alpha = max(0.05, 4**-0.7)
    assert delta == pytest.approx(target - 0.5)
    assert table.q[0, RIGHT] == pytest.approx(0.5 + alpha * (target - 0.5), rel=1e-6)
    assert table.n[0, RIGHT] == 4


def test_a_terminal_transition_does_not_bootstrap() -> None:
    learner = make_learner()
    learner.table.q[5] = 100.0
    learner.update(0, UP, 1.0, None)
    assert learner.table.q[0, UP] == 1.0


def test_the_step_size_decays_polynomially_down_to_its_floor() -> None:
    learner = make_learner()
    assert learner.step_size(0, UP) == 1.0
    learner.table.n[0, UP] = 1
    assert learner.step_size(0, UP) == pytest.approx(2**-0.7)
    learner.table.n[0, UP] = 1_000_000
    assert learner.step_size(0, UP) == 0.05


def test_a_non_terminal_update_needs_the_next_mask() -> None:
    with pytest.raises(ValueError):
        make_learner().update(0, UP, 0.0, 1, [])


# --- selection -------------------------------------------------------------


def test_greedy_selection_takes_the_best_allowed_action() -> None:
    learner = make_learner()
    learner.table.q[0, :3] = [5.0, 1.0, 3.0]  # UP is best but not allowed
    choices = [learner.select(0, [RIGHT, DOWN]) for _ in range(50)]
    assert {choice.action for choice in choices} == {DOWN}
    assert not any(choice.explored for choice in choices)
    assert all(choice.unseen for choice in choices)

    learner.table.n[0, DOWN] = 1
    assert not learner.select(0, [RIGHT, DOWN]).unseen


def test_greedy_ties_are_broken_at_random() -> None:
    picks = {
        make_learner(seed=seed).select(0, [LEFT, BOMB]).action for seed in range(30)
    }
    assert picks == {LEFT, BOMB}


def test_exploration_stays_inside_the_mask() -> None:
    learner = make_learner()
    learner.table.q[0, UP] = 10.0
    choices = [learner.select(0, [RIGHT, WAIT], epsilon=1.0) for _ in range(100)]
    assert {choice.action for choice in choices} == {RIGHT, WAIT}
    assert all(choice.explored for choice in choices)


def test_selection_needs_a_non_empty_mask() -> None:
    with pytest.raises(ValueError):
        make_learner().select(0, [])


def test_learns_the_values_of_a_corridor() -> None:
    """Cells 0..4 in a row; stepping right of cell 4 pays 1 and ends the
    episode. LEFT is masked at cell 0 (a wall). With gamma 0.9 the value of a
    cell k steps from the goal is 0.9 ** (k - 1)."""
    learner = make_learner(gamma=0.9)
    rng = random.Random(7)

    def mask(cell: int) -> list[int]:
        return [RIGHT] if cell == 0 else [LEFT, RIGHT]

    for _ in range(2000):
        cell = rng.randrange(5)
        while True:
            action = learner.select(cell, mask(cell), epsilon=0.3).action
            nxt = cell + 1 if action == RIGHT else cell - 1
            if nxt == 5:
                learner.update(cell, action, 1.0, None)
                break
            learner.update(cell, action, 0.0, nxt, mask(nxt))
            cell = nxt

    for cell in range(5):
        expected = 0.9 ** (5 - cell - 1)
        assert learner.value(cell, mask(cell)) == pytest.approx(expected, abs=1e-2)
        assert learner.select(cell, mask(cell)).action == RIGHT


# --- storage ---------------------------------------------------------------


def random_table(seed: int = 0) -> QTable:
    rng = np.random.default_rng(seed)
    table = QTable.zeros(E1)
    table.q[:] = rng.normal(size=table.q.shape).astype(np.float32)
    table.n[:] = rng.integers(0, 1000, size=table.n.shape, dtype=np.uint32)
    table.meta = {"rounds": 3, "config": {"gamma": 0.99}, "parent": None}
    return table


def test_save_and_load_round_trip_exactly(tmp_path: Path) -> None:
    table = random_table()
    path = tmp_path / "nested" / "q_table.npz"
    table.save(path)
    loaded = QTable.load(path, E1)

    assert np.array_equal(loaded.q, table.q)
    assert np.array_equal(loaded.n, table.n)
    assert loaded.q.dtype == np.float32
    assert loaded.n.dtype == np.uint32
    assert loaded.meta == table.meta
    assert sorted(p.name for p in path.parent.iterdir()) == ["q_table.npz"]


def test_the_file_header_describes_rows_and_columns(tmp_path: Path) -> None:
    path = tmp_path / "q.npz"
    QTable.zeros(E1).save(path)
    with np.load(path, allow_pickle=False) as archive:
        header = json.loads(str(archive["meta"].item()))["header"]
    assert header["schema_id"] == E1.schema_id
    assert header["encoding"] == "E1"
    assert header["actions"] == list(ACTIONS)
    assert header["radices"] == list(E1.radices)


def test_a_table_for_another_encoding_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "q.npz"
    QTable.zeros(E1).save(path)
    with pytest.raises(SchemaMismatchError):
        QTable.load(path, E2)


def test_arrays_of_the_wrong_shape_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "q.npz"
    broken = QTable.zeros(E1)
    broken.q = np.zeros((3, 6), dtype=np.float32)
    broken.save(path)
    with pytest.raises(ValueError):
        QTable.load(path, E1)


def test_a_failed_save_leaves_the_previous_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "q.npz"
    good = QTable.zeros(E1)
    good.q[0, UP] = 1.0
    good.save(path)

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(qtable.np, "savez_compressed", fail)
    with pytest.raises(RuntimeError):
        QTable.zeros(E1).save(path)
    monkeypatch.undo()

    assert QTable.load(path, E1).q[0, UP] == 1.0
    assert [p.name for p in tmp_path.iterdir()] == ["q.npz"]


# --- config ----------------------------------------------------------------


def test_learning_parameters_default_and_override() -> None:
    assert (Config().policy, Config().gamma) == ("learned", 0.99)
    assert (Config().alpha_omega, Config().alpha_min) == (0.7, 0.05)
    config = Config.from_env({ENV_VAR: '{"policy": "random", "gamma": 1}'})
    assert (config.policy, config.gamma) == ("random", 1)


@pytest.mark.parametrize(
    "raw",
    [
        '{"policy": "greedy"}',
        '{"gamma": 0}',
        '{"gamma": 1.5}',
        '{"alpha_min": -0.1}',
        '{"alpha_omega": "0.7"}',
        '{"gamma": true}',
    ],
)
def test_bad_learning_parameters_are_rejected(raw: str) -> None:
    with pytest.raises(ValueError):
        Config.from_env({ENV_VAR: raw})
