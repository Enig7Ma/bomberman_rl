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

import hashlib
import os
import shutil
from pathlib import Path
from typing import Any, Final

from agent_code.tabular_q_agent.features import ENCODINGS
from agent_code.tabular_q_agent.qtable import QTable
from tournament.engine import REPO_ROOT
from training.config import LEARNER
from training.spec import SPECS, AgentSpec

AGENT_CODE: Final = REPO_ROOT / "agent_code"
_NOT_COPIED = shutil.ignore_patterns("logs", "model", "__pycache__", "*.pyc")


def frozen_name(
    run_seed: int, *, spec: AgentSpec = SPECS[LEARNER], namespace: Path | None = None
) -> str:
    """The frozen opponent's directory for one training run."""
    suffix = (
        ""
        if namespace is None
        else "_" + hashlib.sha256(str(namespace.resolve()).encode()).hexdigest()[:12]
    )
    return f"{spec.frozen_prefix}s{run_seed}{suffix}"


def env_prefix(name: str) -> str:
    """The prefix the agent in ``agent_code/<name>`` gives its variables."""
    return name.upper()


def _checked(name: str) -> Path:
    if (
        not name.startswith(tuple(spec.frozen_prefix for spec in SPECS.values()))
        or not name.isidentifier()
    ):
        raise ValueError(f"{name!r} is not a frozen opponent's directory name")
    target = (AGENT_CODE / name).resolve()
    if target.parent != AGENT_CODE.resolve():
        raise ValueError("frozen directory resolves outside agent_code")
    return target


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


def materialise_agent(
    name: str, model: Path | None, *, spec: AgentSpec, params: dict[str, Any], seed: int
) -> Path:
    """Install a snapshot in a run-scoped package, separate from training files."""
    target = _checked(name)
    # Reuse code directories: another world may already have imported this package.
    shutil.copytree(
        AGENT_CODE / spec.name, target, ignore=_NOT_COPIED, dirs_exist_ok=True
    )
    destination = target / "model" / spec.model_file
    destination.parent.mkdir(parents=True, exist_ok=True)
    if model is None:
        spec.initial_model(destination, params, seed)
    else:
        spec.model_info(model, params)  # fail loudly instead of inference fallback
        tmp = destination.with_suffix(".tmp")
        shutil.copyfile(model, tmp)
        os.replace(tmp, destination)
    return destination
