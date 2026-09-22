"""Container-side package verification; standard library only, no training."""

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any


def run(output: Path) -> None:
    # Conda's platform configuration module is stdlib but is not listed in
    # sys.stdlib_module_names. Obtain it from Python itself, not from the agent.
    sysconfig.get_config_vars()
    platform_stdlib = {n for n in sys.modules if n.startswith("_sysconfigdata_")}
    output.mkdir(parents=True, exist_ok=True)
    root = Path("/home/bomberman")
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version,
        "versions": {
            n: importlib.metadata.version(n)
            for n in ("numpy", "pygame", "scipy", "torch", "tensorflow")
        },
        "agents": {},
    }
    for agent, filename in (
        ("dqn_agent", "q_net.npz"),
        ("tabular_q_agent", "q_table.npz"),
    ):
        target = output / agent
        target.mkdir(exist_ok=False)
        checks: dict[str, Any] = {}
        for mode in (
            "framework-imports",
            "default",
            "explicit",
            "missing",
            "corrupt",
            "schema",
            "default-missing",
            "default-corrupt",
            "game",
        ):
            artifact = target / f"{mode}.json"
            command = [
                sys.executable,
                "-I",
                str(root / "d12_probe.py"),
                "--root",
                str(root),
                "--agent",
                agent,
                "--mode",
                mode,
                "--out",
                str(artifact),
            ]
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=180
            )
            (target / f"{mode}.log").write_text(completed.stdout + completed.stderr)
            if completed.returncode:
                raise RuntimeError(f"{agent}/{mode}: {completed.stderr}")
            checks[mode] = json.loads(artifact.read_text())
        allowed = (
            set(checks["framework-imports"]["roots"])
            | set(sys.stdlib_module_names)
            | {"numpy", "agent_code"}
            | platform_stdlib
        )
        for mode in ("default", "explicit", "game"):
            assert set(checks[mode]["roots"]) <= allowed
        model = root / "agent_code" / agent / "model" / filename
        digest = hashlib.sha256(model.read_bytes()).hexdigest()
        assert checks["default"]["loaded_model_sha256"] == digest
        log = (root / "logs/game.log").read_text()
        shutil.copy2(root / "logs/game.log", target / "game_engine.log")
        game = checks["game"]["game"]
        timeouts = log.count(f"Agent <{agent}> exceeded think time")
        skips = log.count(f"Skipping agent <{agent}>")
        result["agents"][agent] = {
            "model_sha256": digest,
            "setup_seconds": checks["default"]["setup_seconds"],
            "game": game,
            "timeouts": timeouts,
            "skips": skips,
            "explicit_rejections": {
                m: checks[m]["rejected"] for m in ("missing", "corrupt", "schema")
            },
            "acceptance": {
                "setup": checks["default"]["setup_seconds"] < 2,
                "p99": game["p99_ms"] < 50,
                "max": game["max_ms"] < 250,
                "timeouts": timeouts == skips == 0,
            },
        }
    (output / "linux_results.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    run(Path(sys.argv[1]))
