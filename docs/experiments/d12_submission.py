"""Install only the frozen winner's tested ZIP, without modifying agent models."""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from docs.experiments.d12_heldout import PROTOCOL, ROOT, sha, write
from docs.experiments.d12_summarize import choose


def package(report: Path, evidence: Path) -> None:
    results = json.loads(report.read_text())
    frozen = json.loads(PROTOCOL.read_text())
    assert results["protocol_sha256"] == sha(PROTOCOL)
    assert results["games"] == frozen["budget_games"]["total"]
    groups = results["summaries"]
    winner, reason = choose(
        groups["dqn_stage2_baseline"]["vs-rule-based"],
        groups["tabular_blended"]["vs-rule-based"],
    )
    assert results["selection"] == {"winner": winner, "reason": reason}
    latency = results["latency"][winner]
    if latency["timeouts"] or latency["p99_ms"] >= 50 or latency["max_ms"] >= 250:
        raise RuntimeError(
            "Selected winner failed latency gate; do not substitute a model"
        )
    candidate = frozen["packages"][winner]
    source = Path(candidate["zip"])
    assert sha(source) == candidate["zip_sha256"]
    with zipfile.ZipFile(source) as archive:
        actual = {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in archive.namelist()
        }
        assert actual == candidate["members"]
    target = ROOT / "final-project-agent-code.zip"
    backup = evidence / "previous_submission.zip"
    if not backup.exists():
        shutil.copy2(target, backup)
    temporary = target.with_suffix(".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(target)
    assert sha(target) == candidate["zip_sha256"]
    write(
        ROOT / "docs/experiments/d12_submission.json",
        {
            "winner": winner,
            "reason": reason,
            "zip": target.name,
            "zip_sha256": sha(target),
            "model_sha256": candidate["model_sha256"],
            "members": candidate["members"],
            "result_sha256": sha(report),
            "protocol_sha256": sha(PROTOCOL),
            "previous_zip_backup": str(backup),
            "previous_zip_sha256": sha(backup),
            "note": "Exact already-Linux-tested frozen ZIP; "
            "source agent models unchanged.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    package(args.report, args.evidence)
