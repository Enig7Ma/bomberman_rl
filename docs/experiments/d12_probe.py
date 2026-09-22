"""Isolated extracted-package check; intentionally imports no repository helpers."""

import argparse
import hashlib
import importlib
import json
import logging
import os
import sys
import time
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast


def run(root: Path, name: str, mode: str) -> dict[str, Any]:
    sys.path.insert(0, str(root.resolve()))
    for key in list(os.environ):
        if key.startswith(("DQN_AGENT_", "TABULAR_Q_AGENT_")):
            del os.environ[key]
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    if mode == "framework-imports":
        importlib.import_module("settings")
        importlib.import_module("events")
        return {"roots": sorted({n.split(".")[0] for n in sys.modules})}
    prefix = "DQN_AGENT" if name == "dqn_agent" else "TABULAR_Q_AGENT"
    os.environ[f"{prefix}_PARAMS"] = json.dumps({"seed": 500, "policy": "learned"})
    model = root / "agent_code" / name / "model"
    model /= "q_net.npz" if name == "dqn_agent" else "q_table.npz"
    original = model.read_bytes()
    sha = hashlib.sha256(original).hexdigest()
    fallback = mode in ("default-missing", "default-corrupt")
    if mode == "default-missing":
        model.unlink()  # only the disposable extracted copy
    elif mode == "default-corrupt":
        model.write_bytes(b"not an npz file")
    if mode == "explicit":
        os.environ[f"{prefix}_MODEL"] = str(model.resolve())
    elif mode == "missing":
        os.environ[f"{prefix}_MODEL"] = str(root / "missing.npz")
    elif mode in ("corrupt", "schema"):
        bad = root / "incompatible.npz"
        if mode == "corrupt":
            bad.write_bytes(b"not an npz file")
        else:
            import numpy as np

            with np.load(model, allow_pickle=False) as saved:
                arrays = {key: saved[key].copy() for key in saved.files}
            meta = json.loads(str(arrays["meta"].item()))
            meta["header"]["schema_id"] = "incompatible"
            arrays["meta"] = np.array(json.dumps(meta))
            np.savez(bad, **arrays)
        os.environ[f"{prefix}_MODEL"] = str(bad.resolve())
    start = time.perf_counter()
    callbacks = importlib.import_module(f"agent_code.{name}.callbacks")
    assert callbacks.__file__ is not None
    assert Path(callbacks.__file__).resolve().is_relative_to(root.resolve())
    imported = time.perf_counter()
    agent = SimpleNamespace(logger=logging.getLogger("package-check"), train=False)
    try:
        callbacks.setup(agent)
    except (ValueError, FileNotFoundError) as error:
        if mode not in ("missing", "corrupt", "schema"):
            raise
        return {"rejected": type(error).__name__, "message": str(error)}
    finally:
        if fallback:
            model.write_bytes(original)
    if mode in ("missing", "corrupt", "schema"):
        raise AssertionError("explicit incompatible model silently accepted")
    setup_seconds = time.perf_counter() - imported
    assert agent.trainer is None and not agent.train
    assert agent.model_file.resolve() == model.resolve()
    if name == "dqn_agent":
        assert (agent.q_function is None) == fallback
    else:
        assert (agent.table.visited_states == 0) == fallback
    import numpy as np

    if not fallback:
        with np.load(model, allow_pickle=False) as saved:
            if name == "dqn_agent":
                for key, value in agent.q_function.arrays.items():
                    np.testing.assert_array_equal(value, saved[key])
            else:
                np.testing.assert_array_equal(agent.table.q, saved["q"])
                np.testing.assert_array_equal(agent.table.n, saved["n"])

    field = np.zeros((17, 17), dtype=np.int64)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    field[2:-1:2, 2:-1:2] = -1
    action = callbacks.act(
        agent,
        {
            "round": 1,
            "step": 1,
            "field": field,
            "self": ("candidate", 0, True, (1, 1)),
            "others": [],
            "bombs": [],
            "coins": [(3, 1)],
            "explosion_map": np.zeros((17, 17)),
            "user_input": None,
        },
    )
    assert action in ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
    assert "torch" not in sys.modules
    modules = sorted(sys.modules)
    outside = [
        n
        for n in modules
        if n.startswith("agent_code.")
        and n != f"agent_code.{name}"
        and not n.startswith(f"agent_code.{name}.")
    ]
    assert not outside, outside
    assert not {"training", "tournament", "dev", "scipy", "torch"} & set(modules)
    result: dict[str, Any] = {
        "setup_seconds": setup_seconds,
        "import_seconds": imported - start,
        "action": action,
        "loaded_model_sha256": None if fallback else sha,
        "fallback_test": fallback,
        "roots": sorted({n.split(".")[0] for n in modules}),
        "modules": modules,
    }
    if mode == "game":
        durations: list[float] = []
        original_act = callbacks.act

        @wraps(original_act)
        def timed(self: object, state: object) -> str:
            began = time.perf_counter()
            selected = str(original_act(self, state))
            durations.append(1000 * (time.perf_counter() - began))
            return selected

        cast(Any, callbacks).act = timed
        main = importlib.import_module("main")
        (root / "logs").mkdir(exist_ok=True)
        started = time.perf_counter()
        main.main(
            [
                "play",
                "--agents",
                name,
                "random_agent",
                "random_agent",
                "random_agent",
                "--no-gui",
                "--n-rounds",
                "3",
                "--seed",
                "500",
                "--save-stats",
                str(root / "game_stats.json"),
                "--log-dir",
                str(root / "logs"),
            ]
        )
        assert durations
        assert "torch" not in sys.modules
        result["game"] = {
            "seconds": time.perf_counter() - started,
            "actions": len(durations),
            "mean_ms": float(np.mean(durations)),
            "p99_ms": float(np.quantile(durations, 0.99)),
            "max_ms": max(durations),
            "stats": json.loads((root / "game_stats.json").read_text()),
        }
    assert hashlib.sha256(model.read_bytes()).hexdigest() == sha
    assert not list(model.parent.glob("*.pt"))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(
        json.dumps(run(args.root, args.agent, args.mode), indent=2), encoding="utf-8"
    )
