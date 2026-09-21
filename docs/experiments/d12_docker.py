"""Reproduce the full Docker build and Linux package checks, no held-out games."""

import argparse
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from docs.experiments.d12_prepare import REPO, framework_copy, sha


def run(out: Path) -> None:
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    frozen = json.loads(
        (REPO / "docs/experiments/d12_frozen_protocol.json").read_text()
    )
    packages = frozen["packages"]
    for package in packages.values():
        assert sha(Path(package["zip"])) == package["zip_sha256"]
    with tempfile.TemporaryDirectory(prefix="bomberman_d12_") as temporary:
        context = Path(temporary) / "context"
        framework_copy(Path(packages["dqn_stage2_baseline"]["zip"]), context)
        with zipfile.ZipFile(packages["tabular_blended"]["zip"]) as archive:
            archive.extractall(context / "agent_code")
        shutil.copy2(REPO / "Dockerfile", context / "Dockerfile")
        shutil.copy2(REPO / "docs/experiments/d12_probe.py", context / "d12_probe.py")
        tag = "bomberman-d12:20260921"
        with (out / "build.log").open("w") as log:
            subprocess.run(
                ["docker", "build", "--progress=plain", "-t", tag, str(context)],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        script = REPO / "docs/experiments/d12_linux.py"
        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--cpus",
            "1",
            "-e",
            "OPENBLAS_NUM_THREADS=1",
            "-e",
            "OMP_NUM_THREADS=1",
            "-e",
            "SDL_VIDEODRIVER=dummy",
            "-e",
            "SDL_AUDIODRIVER=dummy",
            "--mount",
            f"type=bind,source={out},target=/evidence",
            "--mount",
            f"type=bind,source={script},target=/d12_linux.py,readonly",
            tag,
            "python",
            "-I",
            "/d12_linux.py",
            "/evidence",
        ]
        with (out / "run.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        inspection = subprocess.check_output(
            ["docker", "image", "inspect", tag], text=True
        )
        (out / "image.json").write_text(inspection)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
