"""The DQN agent directory must stand alone, and must not need Torch to play.

Same contract as ``test_tabular_q_agent_package.py``, plus the one that is
specific to a neural agent: training uses PyTorch, the submission may not. The
organisers' image installs an unpinned conda ``pytorch`` and importing it costs
about a second of the setup budget, so inference is plain NumPy over a ``.npz``
of weights and ``torch`` must never appear on the import path of ``callbacks``.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parent.parent
AGENT = REPO / "agent_code" / "dqn_agent"
MODEL = AGENT / "model" / "q_net.npz"

ALLOWED_ROOTS = {
    "agent_code",
    "numpy",
    "settings",
    "events",
    "items",
    "fallbacks",
    "pygame",
    "tqdm",
}
# This repository's infrastructure and the training-only dependency.
FORBIDDEN = {"tournament", "training", "dev", "pytest", "scipy", "torch"}

PROBE = """
import json, logging, sys, time
from types import SimpleNamespace

before = set(sys.modules)
start = time.perf_counter()
import agent_code.dqn_agent.callbacks as callbacks
imported = time.perf_counter()

agent = SimpleNamespace(logger=logging.getLogger("package_probe"), train=False)
callbacks.setup(agent)
done = time.perf_counter()
loaded = sorted({name.split(".")[0] for name in set(sys.modules) - before})
print(json.dumps({
    "roots": loaded,
    "modules": sorted(set(sys.modules) - before),
    "import_seconds": imported - start,
    "setup_seconds": done - imported,
    "elapsed": done - start,
    "has_network": agent.q_function is not None,
    "encoder": agent.encoder.name,
    "dim": agent.encoder.dim,
    "policy": agent.config.policy,
}))
"""


def probe(code: str = PROBE) -> dict[str, Any]:
    finished = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": str(REPO),
            "HOME": str(Path.home()),
        },
    )
    result: dict[str, Any] = json.loads(finished.stdout.splitlines()[-1])
    return result


@pytest.fixture(scope="module")
def imported() -> dict[str, Any]:
    return probe()


def test_the_agent_imports_nothing_from_this_repository(
    imported: dict[str, Any],
) -> None:
    roots: list[str] = imported["roots"]
    assert FORBIDDEN.isdisjoint(roots), roots
    for forbidden in ("agent_code.bfs_agent", "agent_code.tabular_q_agent"):
        assert forbidden not in imported["modules"]


def test_playing_never_loads_torch_or_the_teacher(imported: dict[str, Any]) -> None:
    """The learner and the search teacher are training-only; the submitted
    directory carries their code but the play path must not touch it."""
    modules: list[str] = imported["modules"]
    assert "torch" not in imported["roots"]
    for training_only in ("learner", "train", "teacher", "replay", "persistence"):
        name = f"agent_code.dqn_agent.{training_only}"
        assert name not in modules, name


def test_the_agent_only_needs_numpy_and_the_framework(
    imported: dict[str, Any],
) -> None:
    stdlib = set(sys.stdlib_module_names)
    extra = {name for name in imported["roots"] if name not in stdlib}
    framework_probe = (
        "import json, sys\n"
        "import settings\n"
        "roots = sorted({name.split('.')[0] for name in sys.modules})\n"
        "print(json.dumps({'roots': roots}))"
    )
    framework_roots = set(probe(framework_probe)["roots"])
    allowed = ALLOWED_ROOTS | ({"colorama"} & framework_roots)
    assert extra <= allowed, sorted(extra - allowed)


def test_setup_is_fast_enough(imported: dict[str, Any]) -> None:
    """The agent's own work -- geometry and reading a ~100 KB weight file --
    is bounded tightly; the import of numpy and the framework is not, because
    it depends on what else the machine is doing."""
    assert imported["setup_seconds"] < 2.0, imported
    assert imported["elapsed"] < 60.0, imported


@pytest.mark.skipif(not MODEL.exists(), reason="no network shipped yet")
def test_the_shipped_network_loads_and_is_trained(imported: dict[str, Any]) -> None:
    import numpy as np

    assert imported["has_network"] and imported["policy"] == "learned"
    assert MODEL.stat().st_size < 20 * 1024 * 1024
    with np.load(MODEL, allow_pickle=False) as archive:
        meta = json.loads(str(archive["meta"].item()))
        shapes = {name: archive[name].shape for name in ("W0", "W1", "W2")}
    header, trained = meta["header"], meta["meta"]
    assert header["encoder"] == imported["encoder"]
    assert header["layer_sizes"][0] == imported["dim"] == shapes["W0"][1]
    assert header["layer_sizes"][-1] == shapes["W2"][0] == 6
    assert int(trained["transitions"]) > 0 and int(trained["rounds_trained"]) > 0
