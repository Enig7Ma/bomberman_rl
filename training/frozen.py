"""Frozen copies of the learner as training opponents.

Self-play against the table being trained adds non-stationarity before it is
useful (plan §4.4), so the opponent is a *frozen* snapshot instead: a copy of
``agent_code/tabular_q_agent`` under ``agent_code/tabular_frozen_<id>/`` whose
default table is the snapshot. Two things make the copy independent of the
learner:

- it is a separate agent directory, so the framework imports it as a separate
  module, never in training mode;
- the agent names its environment variables after its directory
  (``TABULAR_FROZEN_<ID>_PARAMS``), so it never reads the learner's model path.

Only directories with the ``tabular_frozen_`` prefix are ever created or
deleted here; they are gitignored.
"""

import os
import shutil
from pathlib import Path
from typing import Final

from agent_code.tabular_q_agent.features import ENCODINGS
from agent_code.tabular_q_agent.qtable import QTable
from tournament.engine import REPO_ROOT
from training.config import FROZEN_DIR_PREFIX, LEARNER

AGENT_CODE: Final = REPO_ROOT / "agent_code"
_NOT_COPIED = shutil.ignore_patterns("logs", "model", "__pycache__", "*.pyc")


def frozen_name(run_seed: int) -> str:
    """The frozen opponent's directory for one training run."""
    return f"{FROZEN_DIR_PREFIX}s{run_seed}"


def env_prefix(name: str) -> str:
    """The prefix the agent in ``agent_code/<name>`` gives its variables."""
    return name.upper()


def _checked(name: str) -> Path:
    if not name.startswith(FROZEN_DIR_PREFIX) or not name.isidentifier():
        raise ValueError(f"{name!r} is not a frozen opponent's directory name")
    return AGENT_CODE / name


def materialise_frozen(name: str, table: Path | None, *, encoding: str) -> Path:
    """(Re)create ``agent_code/<name>`` from the learner's code and ``table``.

    Without a table the copy gets an empty one for ``encoding``, which plays
    like the safe-random control.
    """
    target = _checked(name)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(AGENT_CODE / LEARNER, target, ignore=_NOT_COPIED)
    install_table(name, table, encoding=encoding)
    return target


def install_table(name: str, table: Path | None, *, encoding: str) -> None:
    """Point an existing frozen copy at another table, atomically."""
    model = _checked(name) / "model" / "q_table.npz"
    model.parent.mkdir(parents=True, exist_ok=True)
    if table is None:
        QTable.zeros(ENCODINGS[encoding]).save(model)
        return
    tmp = model.with_name(f"{model.name}.tmp")
    shutil.copyfile(table, tmp)
    os.replace(tmp, model)


def remove_frozen(name: str) -> None:
    shutil.rmtree(_checked(name), ignore_errors=True)
