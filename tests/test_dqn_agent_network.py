"""NumPy inference, strict weight files and atomic replacement."""

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from agent_code.dqn_agent import network
from agent_code.dqn_agent.encoder import OneHotE3
from agent_code.dqn_agent.features import Features
from agent_code.dqn_agent.network import (
    ModelFormatError,
    QFunction,
    QNetwork,
    masked_greedy,
)


def test_forward_matches_hand_calculation() -> None:
    net = QNetwork.random(OneHotE3(), 0)
    for array in net.arrays.values():
        array.fill(0)
    net.arrays["W0"][:2, :2] = [[1, 2], [-2, 1]]
    net.arrays["b0"][:2] = [1, 1]
    net.arrays["W1"][:2, :2] = [[3, 4], [-1, 1]]
    net.arrays["b1"][:2] = [-1, 3]
    net.arrays["W2"][:2, :2] = [[2, -1], [-1, -1]]
    net.arrays["b2"][0] = 1
    x = np.zeros(32, dtype=np.float32)
    x[:2] = [2, -1]
    function: QFunction = net
    result = function.values(x)
    # h0 = relu([1,-4]); h1 = relu([2,2]); output = [3,-4,0,0,0,0].
    np.testing.assert_array_equal(
        result, np.array([3, -4, 0, 0, 0, 0], dtype=np.float32)
    )
    assert result.dtype == np.float32
    assert x[:2].tolist() == [2, -1]


def test_initialisation_is_reproducible_and_does_not_touch_global_rng() -> None:
    np_before = np.random.get_state()
    py_before = random.getstate()
    first = QNetwork.random(OneHotE3(), 4)
    second = QNetwork.random(OneHotE3(), 4)
    other = QNetwork.random(OneHotE3(), 5)
    for name, array in first.arrays.items():
        assert array.dtype == np.float32
        np.testing.assert_array_equal(array, second.arrays[name])
        assert not np.array_equal(array, other.arrays[name])
    np_after = np.random.get_state()
    assert np_before[0] == np_after[0] and np_before[2:] == np_after[2:]
    np.testing.assert_array_equal(np_before[1], np_after[1])
    assert random.getstate() == py_before


def test_masked_greedy_ignores_disallowed_values_and_randomises_ties() -> None:
    q = np.array([100, 3, 3, -1, 0, float("nan")], dtype=np.float32)
    assert {masked_greedy(q, (1, 2, 4), random.Random(seed)) for seed in range(30)} == {
        1,
        2,
    }
    assert masked_greedy(q, (3, 4), random.Random(1)) == 4
    with pytest.raises(ValueError):
        masked_greedy(q, (), random.Random(1))
    with pytest.raises(ValueError):
        masked_greedy(q, (5,), random.Random(1))


def test_save_load_is_exact(tmp_path: Path) -> None:
    encoder = OneHotE3()
    first = QNetwork.random(encoder, 7)
    first.meta.update(
        {
            "config": {"seed": 7, "encoder": "onehot_e3"},
            "transitions": 123,
            "updates": 12,
            "rounds_trained": 3,
            "parent_checkpoint": "parent.npz",
            "stage": "test",
            "git_commit": "abc123",
            "extra": [1, "hello", None],
        }
    )
    path = tmp_path / "q_net.npz"
    first.save(path)
    second = QNetwork.load(path, encoder)
    assert first.meta == second.meta and first.header == second.header
    for name, values in first.arrays.items():
        assert values.tobytes() == second.arrays[name].tobytes()
    x = encoder.encode(Features())
    np.testing.assert_array_equal(first.values(x), second.values(x))


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_id", "stale"),
        ("encoder", "dense_v1"),
        ("format_version", 2),
        ("format_version", True),
        ("layer_sizes", [32, 64, 128, 6]),
        ("actions", ["RIGHT", "UP", "DOWN", "LEFT", "WAIT", "BOMB"]),
        ("activation", "tanh"),
        ("dtype", "float64"),
    ],
)
def test_incompatible_headers_are_rejected(
    tmp_path: Path, key: str, value: object
) -> None:
    net = QNetwork.random(OneHotE3(), 0)
    header = {**net.header, key: value}
    path = tmp_path / "bad.npz"
    np.savez(
        path,
        allow_pickle=False,
        **net.arrays,
        meta=json.dumps({"header": header, "meta": net.meta}),
    )
    with pytest.raises(ModelFormatError):
        QNetwork.load(path, OneHotE3())


@pytest.mark.parametrize("name", network.ARRAY_NAMES)
@pytest.mark.parametrize("fault", ["missing", "shape", "dtype", "nan", "inf"])
def test_invalid_weights_are_rejected(tmp_path: Path, name: str, fault: str) -> None:
    net = QNetwork.random(OneHotE3(), 0)
    arrays: dict[str, Any] = dict(net.arrays)
    if fault == "missing":
        del arrays[name]
    elif fault == "shape":
        arrays[name] = np.zeros((1,), dtype=np.float32)
    elif fault == "dtype":
        arrays[name] = arrays[name].astype(np.float64)
    else:
        arrays[name].flat[0] = float(fault)
    path = tmp_path / "bad.npz"
    np.savez(path, **arrays, meta=json.dumps({"header": net.header, "meta": net.meta}))
    with pytest.raises(ModelFormatError):
        QNetwork.load(path, OneHotE3())


@pytest.mark.parametrize(
    "key,value",
    [
        ("config", []),
        ("transitions", -1),
        ("updates", True),
        ("rounds_trained", 1.5),
        ("stage", 3),
        ("parent_checkpoint", []),
        ("git_commit", 2),
        ("extra", float("nan")),
    ],
)
def test_invalid_metadata_is_rejected(tmp_path: Path, key: str, value: object) -> None:
    net = QNetwork.random(OneHotE3(), 0)
    path = tmp_path / "bad.npz"
    np.savez(
        path,
        allow_pickle=False,
        **net.arrays,
        meta=json.dumps(
            {
                "header": net.header,
                "meta": {**net.meta, key: value},
            }
        ),
    )
    with pytest.raises(ModelFormatError):
        QNetwork.load(path, OneHotE3())


@pytest.mark.parametrize("fault", ["write", "replace"])
def test_failed_save_preserves_old_file_and_removes_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    path = tmp_path / "q_net.npz"
    net = QNetwork.random(OneHotE3(), 0)
    net.save(path)
    previous = path.read_bytes()

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("simulated disk failure")

    if fault == "replace":
        monkeypatch.setattr(network.os, "replace", fail)
    else:
        monkeypatch.setattr(network.np, "savez_compressed", fail)
    with pytest.raises(OSError, match="simulated"):
        QNetwork.random(OneHotE3(), 1).save(path)
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]


def test_missing_corrupt_and_non_json_files(tmp_path: Path) -> None:
    path = tmp_path / "missing.npz"
    with pytest.raises(FileNotFoundError):
        QNetwork.load(path, OneHotE3())
    path.write_bytes(b"not an npz")
    with pytest.raises(ValueError):
        QNetwork.load(path, OneHotE3())
    net = QNetwork.random(OneHotE3(), 0)
    np.savez(path, allow_pickle=False, **net.arrays, meta="not json")
    with pytest.raises(ValueError):
        QNetwork.load(path, OneHotE3())
