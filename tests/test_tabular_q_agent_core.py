"""``tabular_q_agent/core`` must stay an exact copy of ``bfs_agent``'s modules.

The submission is a single directory, so the agent cannot import
``agent_code.bfs_agent``; it carries copies instead. These tests fail as soon as
either side changes, so the copies are resynced deliberately rather than drift.
"""

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
BFS = REPO / "agent_code" / "bfs_agent"
CORE = REPO / "agent_code" / "tabular_q_agent" / "core"
COPIED = ("attack.py", "params.py", "planning.py", "safety.py", "world_model.py")


@pytest.mark.parametrize("name", COPIED)
def test_core_module_is_a_byte_identical_copy(name: str) -> None:
    assert (CORE / name).read_bytes() == (BFS / name).read_bytes(), (
        f"core/{name} differs from bfs_agent/{name}; copy it again with "
        f"cp agent_code/bfs_agent/{name} agent_code/tabular_q_agent/core/{name}"
    )


def test_core_holds_only_the_copied_modules() -> None:
    # No callbacks.py in particular: the course's pre-run script uses the first
    # directory in the zip that contains one.
    assert sorted(path.name for path in CORE.glob("*.py")) == sorted(COPIED)
