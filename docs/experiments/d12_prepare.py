"""Build provisional D12 packages and test extracted copies, without held-out games."""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from agent_code.dqn_agent.encoder import OneHotE3
from agent_code.dqn_agent.network import QNetwork
from agent_code.tabular_q_agent.features import ENCODINGS
from agent_code.tabular_q_agent.qtable import QTable

REPO = Path(__file__).resolve().parents[2]
BASELINE = Path(
    "results/dqn/d7_stage2_seed0_20260918/checkpoints/transition_000350346/q_net.npz"
)
BLENDED = Path("agent_code/tabular_q_agent/model/q_table.npz")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_info(path: Path, agent: str) -> dict[str, Any]:
    if agent == "dqn_agent":
        net = QNetwork.load(path, OneHotE3())
        return {"header": net.header, "metadata": net.meta}
    table = QTable.load(path, ENCODINGS["E3"])
    metadata = {k: v for k, v in table.meta.items() if k != "transition_driver"}
    if "transition_driver" in table.meta:
        cursor = table.meta["transition_driver"]
        metadata["transition_driver_summary"] = {
            k: cursor[k] for k in ("format_version", "seed", "origin", "next_stage")
        }
    return {
        "header": table.header(),
        "metadata": metadata,
        "visited_states": table.visited_states,
    }


def inference_sources(agent: str) -> list[Path]:
    """Local relative-import closure; optional training files are not shipped."""
    root = REPO / "agent_code" / agent
    pending = [root / "callbacks.py"]
    found: set[Path] = set()
    while pending:
        path = pending.pop().resolve()
        if path in found:
            continue
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"invalid local import: {path}")
        found.add(path)
        for parent in (path.parent, *path.parent.parents):
            if not parent.is_relative_to(root):
                break
            init = parent / "__init__.py"
            if init.is_file() and init not in found:
                pending.append(init)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ImportFrom) or not node.level:
                continue
            base = path.parent
            for _ in range(node.level - 1):
                base = base.parent
            modules = [node.module] if node.module else [a.name for a in node.names]
            for module in modules:
                local = base.joinpath(*module.split("."))
                target = local.with_suffix(".py")
                pending.append(target if target.is_file() else local / "__init__.py")
    return sorted(found)


def build_package(agent: str, model: Path, target: Path) -> dict[str, Any]:
    if target.exists():
        raise FileExistsError(target)
    model_info(model, agent)  # strict schema and weight validation before packaging
    filename = "q_net.npz" if agent == "dqn_agent" else "q_table.npz"
    entries = {
        p.relative_to(REPO / "agent_code").as_posix(): p.read_bytes()
        for p in inference_sources(agent)
    }
    entries[f"{agent}/model/{filename}"] = model.read_bytes()
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(entries.items()):
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data)
    return {
        "zip": str(target),
        "zip_sha256": sha(target),
        "model_sha256": sha(model),
        "bytes": target.stat().st_size,
        "members": {k: hashlib.sha256(v).hexdigest() for k, v in entries.items()},
    }


def framework_copy(package: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for name in (
        "main.py",
        "agents.py",
        "environment.py",
        "events.py",
        "settings.py",
        "items.py",
        "fallbacks.py",
        "replay.py",
    ):
        shutil.copy2(REPO / name, destination / name)
    shutil.copytree(REPO / "assets", destination / "assets")
    agents = destination / "agent_code"
    agents.mkdir()
    (agents / "__init__.py").write_text("")
    opponent = agents / "random_agent"
    opponent.mkdir()
    for source in (REPO / "agent_code/random_agent").glob("*.py"):
        shutil.copy2(source, opponent / source.name)
    with zipfile.ZipFile(package) as archive:
        for name in archive.namelist():
            if not (agents / name).resolve().is_relative_to(agents.resolve()):
                raise ValueError("unsafe archive member")
        archive.extractall(agents)


def package_probe(root: Path, agent: str, mode: str) -> dict[str, Any]:
    output = root / f"probe_{mode}.json"
    foreign_cwd = root / "unrelated_cwd"
    foreign_cwd.mkdir(exist_ok=True)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(REPO / "docs/experiments/d12_probe.py"),
            "--root",
            str(root.resolve()),
            "--agent",
            agent,
            "--mode",
            mode,
            "--out",
            str(output.resolve()),
        ],
        cwd=root if mode == "game" else foreign_cwd,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=180,
    )
    (root / f"probe_{mode}.log").write_text(
        result.stdout + result.stderr, encoding="utf-8"
    )
    if result.returncode:
        raise RuntimeError(f"{mode} failed: {result.stderr}")
    return json.loads(output.read_text(encoding="utf-8"))


def run(out: Path) -> dict[str, Any]:
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    candidates: list[dict[str, Any]] = []
    chosen = [
        (
            "dqn_stage2_baseline",
            "dqn_agent",
            REPO / BASELINE,
            "D6 seed0 -> baseline stage2; manifest and continuation.json retained",
        ),
        (
            "tabular_blended",
            "tabular_q_agent",
            REPO / BLENDED,
            "Maria Druz commit40b5718; heuristic blend delta0.05 in metadata; "
            "full blending script/history not supplied",
        ),
        (
            "tabular_previous",
            "tabular_q_agent",
            REPO / "results/dqn/candidate_backups_20260919/tabular_q_table.npz",
            "Prior deployed table507662e; historical practical validation only",
        ),
    ]
    study = json.loads((REPO / "docs/experiments/d8_reduced_results.json").read_text())
    for curve in study["curves"]:
        if curve["milestone"] != 4:
            continue
        agent, seed = curve["agent"], curve["seed"]
        name = "q_net.npz" if agent == "dqn_agent" else "q_table.npz"
        chosen.append(
            (
                f"d8_{agent}_seed{seed}",
                agent,
                Path(curve["checkpoint"]) / name,
                "Matched reduced D8, fresh seed, navigation50k + crates300k",
            )
        )
    packages: dict[str, Any] = {}
    for identifier, agent, path, origin in chosen:
        record = {
            "id": identifier,
            "agent": agent,
            "source": str(path),
            "sha256": sha(path),
            "origin": origin,
            **model_info(path, agent),
        }
        candidates.append(record)
        if identifier not in ("dqn_stage2_baseline", "tabular_blended"):
            continue
        backup = out / "weights_backup" / identifier / path.name
        backup.parent.mkdir(parents=True)
        shutil.copy2(path, backup)
        archive = build_package(agent, path, out / f"{identifier}.zip")
        checks: dict[str, Any] = {}
        with tempfile.TemporaryDirectory(prefix="d12_package_") as temporary:
            root = Path(temporary) / "framework"
            framework_copy(Path(archive["zip"]), root)
            try:
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
                    checks[mode] = package_probe(root, agent, mode)
            finally:
                # Preserve diagnostics; don't add disposable framework sources to
                # the repository's lint/type-check input or alter its exclusions.
                logs = out / identifier
                logs.mkdir()
                for pattern in ("*.json", "*.log"):
                    for file in root.rglob(pattern):
                        target = logs / file.relative_to(root)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(file, target)
        allowed = (
            set(checks["framework-imports"]["roots"])
            | set(sys.stdlib_module_names)
            | {"numpy", "agent_code"}
        )
        for mode in ("default", "explicit", "game"):
            assert set(checks[mode]["roots"]) <= allowed
            assert checks[mode]["loaded_model_sha256"] == record["sha256"]
        assert sha(path) == sha(backup) == record["sha256"]
        packages[identifier] = {**archive, "checks": checks, "backup": str(backup)}
    result = {
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "status": "provisional packages; candidate decision and held-out pending",
        "candidates": candidates,
        "packages": packages,
    }
    (out / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def summarize(raw: Path, output: Path) -> None:
    """Export compact evidence from existing checks; never replay any games."""
    result = json.loads((raw / "manifest.json").read_text())
    summary: dict[str, Any] = {
        "code_base": result["head"],
        "status": result["status"],
        "raw_directory": str(raw),
        "python": sys.version,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pygame", "pytest", "ruff", "pyright")
        },
        "script_sha256": {
            name: sha(REPO / "docs/experiments" / name)
            for name in ("d12_prepare.py", "d12_probe.py")
        },
        "candidates": result["candidates"],
        "packages": {},
        "held_out_games": 0,
        "technical_rounds": 6,
        "protocol": {
            "scenario": "classic",
            "opponents": ["random_agent"] * 3,
            "rounds_per_agent": 3,
            "world_rng_seed": 500,
            "jobs": 1,
            "note": "Single world RNG stream, not three independently seeded maps; "
            "stock random opponents unmodified, world seed is not full RNG control",
        },
    }
    for identifier, package in result["packages"].items():
        agent = "dqn_agent" if identifier.startswith("dqn") else "tabular_q_agent"
        log = (raw / identifier / "logs/game.log").read_text()
        checks = package["checks"]
        game = checks["game"]["game"]
        timeouts = log.count(f"Agent <{agent}> exceeded think time")
        skips = log.count(f"Skipping agent <{agent}>")
        assert sha(Path(package["zip"])) == package["zip_sha256"]
        compact = {k: v for k, v in package.items() if k != "checks"}
        compact.update(
            default_setup_seconds=checks["default"]["setup_seconds"],
            explicit_setup_seconds=checks["explicit"]["setup_seconds"],
            numpy_only=True,
            explicit_rejections={
                k: checks[k]["rejected"] for k in ("missing", "corrupt", "schema")
            },
            default_fallbacks={
                k: checks[k]["fallback_test"]
                for k in ("default-missing", "default-corrupt")
            },
            game={**game, "timeouts": timeouts, "skipped_actions": skips},
            acceptance={
                "setup_under_2s": checks["default"]["setup_seconds"] < 2,
                "p99_under_50ms": game["p99_ms"] < 50,
                "max_under_250ms": game["max_ms"] < 250,
                "no_timeouts": timeouts == skips == 0,
            },
        )
        summary["packages"][identifier] = compact
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summarize", type=Path)
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize, args.out)
    else:
        run(args.out)
