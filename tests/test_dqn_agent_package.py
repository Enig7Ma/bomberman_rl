"""D12 staging: extracted inference closures work without sibling agents or Torch."""

import sys
import zipfile
from pathlib import Path

import pytest

from agent_code.dqn_agent.encoder import OneHotE3
from agent_code.dqn_agent.network import QNetwork
from agent_code.tabular_q_agent.features import ENCODINGS
from agent_code.tabular_q_agent.qtable import QTable
from docs.experiments.d12_prepare import (
    build_package,
    framework_copy,
    package_probe,
)


@pytest.mark.parametrize("agent", ["dqn_agent", "tabular_q_agent"])
def test_extracted_package_loads_acts_and_rejects_wrong_models(
    tmp_path: Path, agent: str
) -> None:
    model = tmp_path / "source.npz"
    if agent == "dqn_agent":
        QNetwork.random(OneHotE3(), 10).save(model)
    else:
        table = QTable.zeros(ENCODINGS["E3"])
        table.n[0, 0] = 1
        table.save(model)
    package = tmp_path / "agent.zip"
    archive = build_package(agent, model, package)
    with zipfile.ZipFile(package) as saved:
        names = saved.namelist()
    assert f"{agent}/callbacks.py" in names
    assert sum(n.endswith("callbacks.py") for n in names) == 1
    assert not any(
        x in n
        for n in names
        for x in ("logs/", "checkpoint", "replay.npz", "train.py", "__pycache__")
    )
    root = tmp_path / "framework"
    framework_copy(package, root)
    baseline = set(package_probe(root, agent, "framework-imports")["roots"])
    for mode in ("default", "explicit"):
        checked = package_probe(root, agent, mode)
        assert checked["loaded_model_sha256"] == archive["model_sha256"]
        assert checked["setup_seconds"] < 2
        assert set(checked["roots"]) <= baseline | set(sys.stdlib_module_names) | {
            "numpy",
            "agent_code",
        }
    for mode in ("missing", "corrupt", "schema"):
        assert package_probe(root, agent, mode)["rejected"]
    for mode in ("default-missing", "default-corrupt"):
        assert package_probe(root, agent, mode)["fallback_test"]
    # Deterministic bytes make package identity independently reproducible.
    assert (
        build_package(agent, model, tmp_path / "second.zip")["zip_sha256"]
        == (archive["zip_sha256"])
    )
