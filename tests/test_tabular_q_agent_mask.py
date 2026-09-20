"""Tests for ``tabular_q_agent``'s action mask and configuration."""

import pytest

from agent_code.tabular_q_agent.config import ENV_VAR, Config, MaskVariant
from agent_code.tabular_q_agent.core.safety import Assessment, Outcome
from agent_code.tabular_q_agent.mask import allowed_actions


def graded(action: str, tier: int, survived: int = 5) -> Assessment:
    """An assessment whose ``tier`` property comes out as ``tier``."""

    def outcome(ok: bool) -> Outcome:
        return Outcome(survives=ok, survived=5 if ok else survived, refuges=int(ok))

    return Assessment(
        action=action,
        optimistic=outcome(tier >= 1),
        static=outcome(tier >= 2),
        threatened=outcome(tier >= 3),
        contested=outcome(tier >= 4),
    )


MIXED = [
    graded("UP", 4),
    graded("RIGHT", 3),
    graded("DOWN", 2),
    graded("LEFT", 1),
    graded("WAIT", 0),
    graded("BOMB", 4),
]


def test_graded_builds_the_requested_tier() -> None:
    assert [a.tier for a in MIXED] == [4, 3, 2, 1, 0, 4]


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("best_tier", ["UP", "BOMB"]),
        ("min_tier_2", ["UP", "RIGHT", "DOWN", "BOMB"]),
        ("any_escape", ["UP", "RIGHT", "DOWN", "LEFT", "BOMB"]),
        ("legal", ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]),
    ],
)
def test_each_variant_keeps_its_tiers(
    variant: MaskVariant, expected: list[str]
) -> None:
    assert allowed_actions(MIXED, variant) == expected


@pytest.mark.parametrize("variant", ["best_tier", "min_tier_2", "any_escape"])
def test_without_an_escape_every_variant_plays_for_time(variant: MaskVariant) -> None:
    doomed = [graded("UP", 0, survived=2), graded("WAIT", 0, survived=3)]
    assert allowed_actions(doomed, variant) == ["WAIT"]


def test_default_config() -> None:
    assert Config() == Config.from_env({})
    assert Config().mask == "best_tier"
    assert Config().seed is None


def test_env_overrides_fields() -> None:
    config = Config.from_env({ENV_VAR: '{"mask": "legal", "seed": 7}'})
    assert config == Config(mask="legal", seed=7)


def test_blank_env_means_defaults() -> None:
    assert Config.from_env({ENV_VAR: "  "}) == Config()


@pytest.mark.parametrize(
    "raw",
    [
        '{"temperature": 0.9}',  # unknown field
        "[1, 2]",  # not an object
        '{"mask": "safest"}',  # not a mask variant
        '{"seed": "0"}',  # not an integer
        '{"seed": true}',  # bool is not a seed
    ],
)
def test_bad_overrides_are_rejected(raw: str) -> None:
    with pytest.raises(ValueError):
        Config.from_env({ENV_VAR: raw})
