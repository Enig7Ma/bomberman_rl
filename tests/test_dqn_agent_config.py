"""Scaffold configuration, including paths under the framework's chdir."""

import json
from pathlib import Path

import pytest

from agent_code.dqn_agent.config import (
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    ENV_VAR,
    METRICS_ENV_VAR,
    MODEL_ENV_VAR,
    REPO_ROOT,
    Config,
    metrics_path,
    model_path,
)
from agent_code.dqn_agent.encoder import ENCODERS


def test_defaults_and_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    assert Config.from_env({}) == Config()
    assert Config.from_env({ENV_VAR: "  "}) == Config()
    assert Config().policy == "learned"
    assert Config().encoding == "E3"
    monkeypatch.setenv(ENV_VAR, '{"mask": "legal", "policy": "learned", "seed": 4}')
    assert Config.from_env() == Config(mask="legal", policy="learned", seed=4)


@pytest.mark.parametrize(
    "raw",
    [
        "[]",
        "null",
        "broken",
        '{"unknown": 1}',
        '{"encoding": "E1"}',
        '{"mask": "unsafe"}',
        '{"policy": "heuristic"}',
        '{"seed": true}',
        '{"seed": 1.5}',
        '{"seed": "2"}',
        '{"init_seed": true}',
        '{"init_seed": -1}',
        '{"init_seed": 1.5}',
        '{"init_seed": "2"}',
    ],
)
def test_invalid_configuration_is_rejected(raw: str) -> None:
    with pytest.raises(ValueError):
        Config.from_env({ENV_VAR: raw})


def test_tabular_environment_does_not_configure_dqn() -> None:
    assert Config.from_env({"TABULAR_Q_AGENT_PARAMS": '{"seed": 7}'}) == Config()


def test_d3_defaults() -> None:
    c = Config()
    assert (c.gamma, c.lr, c.batch_size, c.grad_clip) == (0.99, 3e-4, 64, 10)
    assert (c.replay_size, c.warmup, c.train_every, c.target_every) == (
        100_000,
        5000,
        4,
        1000,
    )
    assert (c.epsilon_start, c.epsilon_end, c.epsilon_fraction) == (0.3, 0.05, 0.6)
    assert (c.c_coin, c.crate_aid, c.death_aid) == (0.5, 0, 0)
    assert (
        Config.from_env({ENV_VAR: '{"lr": 0.001, "target_every": 50}'}).target_every
        == 50
    )


@pytest.mark.parametrize(
    "name,value",
    [
        ("gamma", -1),
        ("gamma", 1.1),
        ("lr", 0),
        ("lr", float("nan")),
        ("grad_clip", 0),
        ("batch_size", True),
        ("batch_size", 1.5),
        ("replay_size", 0),
        ("warmup", -1),
        ("train_every", 0),
        ("target_every", 0),
        ("c_coin", float("inf")),
        ("death_aid", "bad"),
        ("epsilon_start", 1.1),
        ("epsilon_end", -0.1),
        ("epsilon_end", 0.9),
        ("epsilon_fraction", 0),
        ("epsilon_fraction", 1.1),
    ],
)
def test_d3_invalid_config(name: str, value: object) -> None:
    with pytest.raises(ValueError):
        Config.from_env({ENV_VAR: json.dumps({name: value})})


@pytest.mark.parametrize("seed", [None, 0, 123])
def test_init_seed(seed: int | None) -> None:
    assert Config.from_env({ENV_VAR: json.dumps({"init_seed": seed})}).init_seed == seed


def test_paths_are_independent_of_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert model_path({}) == (DEFAULT_MODEL_PATH, False)
    assert model_path({MODEL_ENV_VAR: " "}) == (DEFAULT_MODEL_PATH, False)
    assert metrics_path({}) == DEFAULT_METRICS_PATH
    relative = {
        MODEL_ENV_VAR: "results/dqn/q_net.npz",
        METRICS_ENV_VAR: "results/dqn/metrics.jsonl",
    }
    assert model_path(relative) == (REPO_ROOT / relative[MODEL_ENV_VAR], True)
    assert metrics_path(relative) == REPO_ROOT / relative[METRICS_ENV_VAR]
    model, metrics = tmp_path / "net.npz", tmp_path / "metrics.jsonl"
    assert model.is_absolute() and metrics.is_absolute()
    monkeypatch.setenv(MODEL_ENV_VAR, str(model))
    monkeypatch.setenv(METRICS_ENV_VAR, str(metrics))
    assert model_path() == (model, True)
    assert metrics_path() == metrics
    assert model_path({MODEL_ENV_VAR: "~/net.npz"}) == (Path.home() / "net.npz", True)


@pytest.mark.parametrize("mask", ["best_tier", "min_tier_2", "any_escape", "legal"])
def test_every_shared_mask_is_supported(mask: str) -> None:
    assert Config.from_env({ENV_VAR: json.dumps({"mask": mask})}).mask == mask


def test_encoder_defaults_and_environment_override() -> None:
    assert Config().encoder == "onehot_e3"
    parsed = Config.from_env({ENV_VAR: '{"encoder": "onehot_e3"}'})
    assert ENCODERS[parsed.encoder].dim == 32


@pytest.mark.parametrize("value", ["missing", "dense_v1", "", None, 1, [], {}])
def test_unknown_or_invalid_encoder_is_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="encoder"):
        Config.from_env({ENV_VAR: json.dumps({"encoder": value})})
