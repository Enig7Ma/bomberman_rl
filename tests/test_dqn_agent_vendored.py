"""Shared modules stay identical and the submission imports independently."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from agent_code.dqn_agent.config import Config
from agent_code.dqn_agent.encoder import ENCODERS
from agent_code.dqn_agent.network import QNetwork

REPO = Path(__file__).resolve().parent.parent
TABULAR = REPO / "agent_code" / "tabular_q_agent"
DQN = REPO / "agent_code" / "dqn_agent"
SHARED = (
    "features.py",
    "mask.py",
    "symmetry.py",
    "rewards.py",
    "bookkeeping.py",
    "core/world_model.py",
    "core/safety.py",
    "core/planning.py",
    "core/attack.py",
    "core/params.py",
)


@pytest.mark.parametrize("name", SHARED)
def test_shared_modules_are_byte_identical(name: str) -> None:
    assert (DQN / name).read_bytes() == (TABULAR / name).read_bytes()


def test_core_contains_only_shared_modules() -> None:
    assert {p.name for p in (DQN / "core").glob("*.py")} == {
        Path(name).name for name in SHARED if name.startswith("core/")
    }
    assert list(DQN.glob("*/callbacks.py")) == []


@pytest.mark.parametrize("loaded", [False, True])
def test_agent_works_without_other_agents_or_training(
    tmp_path: Path, loaded: bool
) -> None:
    # Copy only the submission and the framework modules it is allowed to need.
    # ``model`` is left out so the two cases are "no weights at all" and
    # "weights named explicitly", independently of what the repository ships.
    destination = tmp_path / "agent_code" / "dqn_agent"
    shutil.copytree(
        DQN,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "logs", "model"),
    )
    for name in ("settings.py", "fallbacks.py", "events.py"):
        shutil.copyfile(REPO / name, tmp_path / name)
    script = """
import json, logging, sys
from types import SimpleNamespace
import numpy as np
from agent_code.dqn_agent import callbacks, bookkeeping, rewards, symmetry
agent = SimpleNamespace(logger=logging.getLogger('probe'), train=False)
callbacks.setup(agent)
field = np.full((5, 5), -1, dtype=np.int64)
field[1:4, 1:4] = 0
action = callbacks.act(agent, {
    'round': 1, 'step': 1, 'field': field,
    'self': ('probe', 0, True, (1, 1)), 'others': [], 'bombs': [], 'coins': [],
    'explosion_map': np.zeros_like(field, dtype=np.float64), 'user_input': None,
})
print(json.dumps({'action': action, 'loaded': agent.q_function is not None,
                  'modules': sorted(sys.modules)}))
"""
    env = {k: v for k, v in os.environ.items() if not k.startswith("DQN_AGENT_")}
    env["PYTHONPATH"] = str(tmp_path)
    path = tmp_path / "random.npz"
    if loaded:
        QNetwork.random(ENCODERS[Config().encoder], 0).save(path)
        env["DQN_AGENT_MODEL"] = str(path)
    before = (path.read_bytes(), path.stat().st_mtime_ns) if loaded else None
    finished = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert not (tmp_path / "checkpoint.pt").exists()
    assert not (tmp_path / "replay.npz").exists()
    if loaded:
        assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    result = json.loads(finished.stdout.splitlines()[-1])
    assert result["action"] in {"RIGHT", "DOWN", "WAIT", "BOMB"}
    assert result["loaded"] is loaded
    modules: list[str] = result["modules"]
    roots = {name.split(".")[0] for name in modules}
    assert roots.isdisjoint({"torch", "training", "tournament", "dev", "scipy"})
    assert all(
        not name.startswith("agent_code.") or name.startswith("agent_code.dqn_agent")
        for name in modules
    )
