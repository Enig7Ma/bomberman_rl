"""The submitted agent directory must stand alone (plan step Q12).

The organisers copy one directory into their own checkout of the framework and
run it with ``self.train = False``. So ``callbacks`` may only pull in the
standard library, numpy and the framework's own modules -- never this
repository's ``tournament``, ``training`` or ``dev`` code, and never
``bfs_agent`` (its code is *copied* into ``core/``).

The table itself is checked once it ships; until then those tests skip, so the
suite stays green while Q12 is still open.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parent.parent
AGENT = REPO / "agent_code" / "tabular_q_agent"
MODEL = AGENT / "model" / "q_table.npz"

# Everything the agent may import beyond the standard library. ``pygame`` and
# ``tqdm`` are not the agent's doing: ``settings`` pulls them in through
# ``fallbacks``, and the organisers' image ships both.
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
# This repository's own infrastructure, which a submission must never need.
FORBIDDEN = {"tournament", "training", "dev", "pytest", "scipy"}

PROBE = """
import json, logging, sys, time
from types import SimpleNamespace

before = set(sys.modules)
start = time.perf_counter()
import agent_code.tabular_q_agent.callbacks as callbacks
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
    "visited_states": agent.table.visited_states,
    "encoding": agent.encoding.name,
    "policy": agent.config.policy,
}))
"""


# Measure framework dependencies separately so an agent-only import cannot
# grant itself an exception to ALLOWED_ROOTS.
FRAMEWORK_PROBE = """
import json, sys
import settings
print(json.dumps({"roots": sorted({name.split(".")[0] for name in sys.modules})}))
"""


def probe(code: str = PROBE) -> dict[str, Any]:
    """Run an import probe in a fresh interpreter and report what it loaded."""
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
    assert "agent_code.bfs_agent" not in imported["modules"]
    outside = [name for name in roots if name in {"tournament", "training", "dev"}]
    assert not outside


def test_the_agent_only_needs_numpy_and_the_framework(
    imported: dict[str, Any],
) -> None:
    # Third-party roots the agent pulls in, ignoring the standard library.
    stdlib = set(sys.stdlib_module_names)
    extra = {name for name in imported["roots"] if name not in stdlib}
    # settings -> fallbacks -> tqdm -> colorama on Windows. Only allow this
    # transitive dependency when the framework itself actually imports it.
    framework_roots = set(probe(FRAMEWORK_PROBE)["roots"])
    allowed = ALLOWED_ROOTS | ({"colorama"} & framework_roots)
    assert extra <= allowed, sorted(extra - allowed)


def test_setup_is_fast_enough(imported: dict[str, Any]) -> None:
    """Bound the agent's own work tightly, the imports only loosely.

    ``setup`` builds the wall geometry and loads the table; that is what the
    agent controls. Importing numpy and the framework dominates the total and
    depends on the machine: measured at 2.7-8.2 s while three training jobs
    were running (load average 32 on 8 cores), against ~0.5 s idle. Bounding
    the total tightly would make this test fail whenever training runs beside
    it, which says nothing about the submission.
    """
    assert imported["setup_seconds"] < 2.0, imported
    assert imported["elapsed"] < 60.0, imported


@pytest.mark.skipif(not MODEL.exists(), reason="no Q-table shipped yet (plan step Q12)")
def test_the_shipped_table_is_trained(imported: dict[str, Any]) -> None:
    assert imported["visited_states"] > 0
    assert imported["policy"] == "learned"
    assert MODEL.stat().st_size < 20 * 1024 * 1024


@pytest.mark.skipif(not MODEL.exists(), reason="no Q-table shipped yet (plan step Q12)")
def test_the_shipped_table_matches_the_default_encoding(
    imported: dict[str, Any],
) -> None:
    import numpy as np

    with np.load(MODEL, allow_pickle=False) as archive:
        header = json.loads(str(archive["meta"].item()))["header"]
    assert header["encoding"] == imported["encoding"]
