"""Run held-out harness inside the already tested immutable Linux image."""

import argparse
import json
import subprocess
from pathlib import Path

from docs.experiments.d12_heldout import ROOT


def execute(out: Path, mode: str) -> None:
    out = out.resolve()
    lock = json.loads((ROOT / "docs/experiments/d12_runtime_lock.json").read_text())
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--cpus",
        "1" if mode == "latency" else "2",
    ]
    for key, value in {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "SDL_VIDEODRIVER": "dummy",
        "SDL_AUDIODRIVER": "dummy",
        "PYGAME_HIDE_SUPPORT_PROMPT": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }.items():
        command += ["-e", f"{key}={value}"]
    for source, target, readonly in [
        (ROOT / "docs", "/home/bomberman/docs", True),
        (ROOT / "training", "/home/bomberman/training", True),
        (ROOT / "tournament", "/home/bomberman/tournament", True),
        (out / "models", "/models", True),
        (out, "/evidence", False),
    ]:
        command += [
            "--mount",
            f"type=bind,source={source},target={target}"
            + (",readonly" if readonly else ""),
        ]
    command += [
        lock["image"]["Id"],
        "python",
        "-m",
        "docs.experiments.d12_heldout",
        "--out",
        f"/evidence/{mode}",
        "--models",
        "/models",
        "--mode",
        mode,
    ]
    (out / f"{mode}_command.json").write_text(json.dumps(command, indent=2))
    with (out / f"{mode}.log").open("a", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    (out / f"{mode}_exit.json").write_text(
        json.dumps({"returncode": result.returncode})
    )
    result.check_returncode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("smoke", "heldout", "latency"), required=True
    )
    args = parser.parse_args()
    execute(args.out, args.mode)
