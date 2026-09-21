# D12: blended validation, candidate freeze and Linux verification

2026-09-21. Authorized scope:25 blended-table validation games, freeze candidates
and final protocol, Docker build/package checks. No held-out games, submission
winner, PR, commit, push or merge in this task. No training or parameter changes.
Existing uncommitted D12 preparation and all earlier experiments are preserved.

## Blended-table validation

Only25 new rounds were played: classic vs3 rule_based_agent, seeds500–524,
candidate seat0, normal seeded starting-position permutation,400-step limit,
jobs1, train=False, epsilon0, learned policy, strict explicit model. Candidate
private RNG500; rule streams10*world_seed+seat, with setup entropy isolated.
All25 initial board/coin/position hashes match the previous comparison. Exact
loaded q/n arrays match the blended model; source weights remained unchanged.
The previously collected25 DQN and25 reference rounds were reused, not replayed.

| Candidate | Raw score | Delta reference [map95% CI] | Coins | Credited kills | Suicides/25 | Survival/25 | Invalid | Timeouts |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Blended table | 5.84 | +3.08 [1.64,4.48] | 4.04 | 0.36 | 0 | 25 | 153 | 0 |
| Baseline DQN stage2 | 3.20 | +0.44 [-0.76,1.64] | 2.00 | 0.24 | 1 | 23 | 11 | 0 |
| Rule reference | 2.76 | 0 | 2.56 | 0.04 | 18 | 6 | 149 | 0 |

Blended-minus-DQN=+2.64[1.16,4.08]. New games took30.38s. Intervals resample25
paired maps,20000 draws with seed20260921; they do not represent independent
training variability. All intervals use the same bootstrap seed20260921.
Exact per-map results and source checksums:
[d12_blended_validation.json](d12_blended_validation.json).

The blend leads this validation sample. This is practical strength comparison,
not strict D8: budgets/rewards/history differ and the table has a heuristic
blend. Zero suicides on25 maps is not a safety guarantee. Its153 invalid actions
remain in the results; this task does not diagnose or suppress them. The old
table's4.60 score is not attributed to this new model.

## Frozen candidates and protocol

[d12_frozen_protocol.json](d12_frozen_protocol.json) fixes identities, model and
ZIP SHA256, package member hashes, schema/config/metadata and eligibility.
The script rejects overwriting already-created freeze artifacts.

- Eligible DQN: baseline stage2 seed0,350346 transitions,86337 updates,
  n_step1,lr3e-4,target_every1000,onehot_e3/best_tier. Model SHA256:
  `fa724eb4ebebefd2170cb15afe18289e79d023743916252c5ba4238d687c7f33`.
- Eligible table: Maria's commit40b5718 blended model, declared delta0.05.
  Model SHA256: `ffd992c6e5c8236d1c012d31bf43b7ed737542cdc19ab8965702c1cd3afeef2d`.
- All six matched D8 finals are frozen as a separate diagnostic seed panel.
  They cannot become a best-seed submission choice after seeing held-out scores.
  The older deployed table stays a historical control, outside final selection.

The already-tested ZIPs remain byte-identical in
`results/dqn/d12_preparation_20260921_v2/`; no default repository model is
replaced. Full baseline checkpoint/replay remain at their original location.
NumPy weight backups are retained alongside the ZIPs. Full blending recipe and
independent training history are still unavailable; metadata is not embellished.

The user confirmed that Q12 is unavailable. Final operational protocol therefore
uses explicit repository presets: primary vs-rule-based; secondary vs-peaceful,
vs-coin-collector,mixed,coin-heaven-solo,crates-solo. Seeds3000–3099, all four
candidate seats in combat and one in solo. Same private opponent streams and
board hashes across paired arms, strict NumPy inference, no fallback/updates.

Eight frozen models x1800 games,1800 shared reference games and1200 head-to-head
games = **17400 future games**. Head-to-head uses50 seeds3000–3049, all24
permutations of two candidates and two labelled rule opponents; private stream
identity distinguishes the two rules. Bootstrap clusters all seats/permutations
by world seed; D8 panels additionally resample training seeds outside maps.
Two processes for evaluation, one for latency. Failed/limited games remain.

Selection is between the two practical candidates only, by the original D12
rule: larger paired delta to rule reference; overlapping delta intervals -> fewer
primary-game suicides; remaining tie -> table. No winner is chosen now. No
future schedule was executed here; the entire17400-game budget belongs to the
next authorized task, not these three steps.

## Docker build changes

The original floating Miniconda base selected Python3.14; TensorFlow had no
matching distribution and the build failed. Pinning3.12 in base then encountered
Anaconda channel ToS and base-package ABI constraints. These failures/logs are
preserved in `results/dqn/d12_final_20260921/`.

Dockerfile now creates an isolated `/opt/bomberman` Python3.12 environment from
conda-forge and uses it via PATH. PyTorch/torchvision use CPU packages. All
original software groups remain, including TensorFlow, SciPy, scikit-learn,
pygame and the optional third-party stack. No terms were accepted on the user's
behalf. The base environment is not downgraded; agent code/weights are unchanged.

The build context is a temporary copy of unchanged framework/assets and the two
frozen packages, excluding results/, replay/checkpoints and .git. Linux checks
run with network disabled, one CPU, one BLAS thread, fresh isolated Python
processes, default/explicit/broken/schema loading and three technical rounds per
candidate vs3 random agents on world seed500. These are technical smoke games,
not strength estimates or held-out games. Package NumPy-only dependency checks
remain meaningful even though the image contains Torch/TensorFlow.

## Reproduction

```powershell
# Fresh directory, exactly25 new validation rounds; don't repeat existing data.
.venv/Scripts/python.exe -m docs.experiments.dqn_candidates --agent tabular_q_agent --reference-results docs/experiments/dqn_candidates_validation.json --out results/dqn/d12_blended_validation_repeat
# Offline evidence/freeze generation; refuses existing frozen output files.
.venv/Scripts/python.exe -m docs.experiments.d12_freeze
# Build minimal context + full Docker stack + Linux checks, no held-out games.
.venv/Scripts/python.exe -m docs.experiments.d12_docker --out results/dqn/d12_docker_repeat
```

For subsequent verification use the recorded immutable image ID, not merely a
mutable tag. Rebuilding against floating dependency repositories can change the
image; the runtime lock records the actual tested image and package versions.

## Completed Linux verification

The full image built successfully. Immutable image ID:
`sha256:0cc76dde373c670a5d185ca79826ece4bc48089cf6b6a68e6f4e0f527b3670a3`.
Actual runtime: Linux amd64 / WSL2, Python3.12.14, NumPy2.5.3,
pygame2.6.1, SciPy1.18.1. Torch2.6.0 and TensorFlow2.21.0 are installed
in the image but not imported by the agents' inference path.
[d12_runtime_lock.json](d12_runtime_lock.json) records image identity,
protocol/source hashes, versions and complete compact Linux measurements.

| Package | Setup ms | Actions | Decision p99 ms | Maximum ms | Timeouts / skips |
|---|---:|---:|---:|---:|---:|
| DQN baseline stage2 | 2.131 | 1200 | 0.647 | 3.011 | 0 / 0 |
| Blended table | 62.684 | 958 | 0.768 | 3.005 | 0 / 0 |

Both pass setup<2s, p99<50ms and max<250ms on this computer. Default and
explicit loading use the expected weights; explicit missing/corrupt/schema
models are rejected. Default-path fallback checks also pass. Fresh processes
verify the NumPy/framework dependency boundary and absence of Torch. All frozen
source models and ZIP checksums were rechecked after execution and are unchanged.

The first Linux checker reached three DQN smoke rounds, then rejected Conda's
standard `_sysconfigdata__linux_x86_64-linux-gnu` module. Python does not include
that generated module in `sys.stdlib_module_names`. The checker now obtains
platform modules from `sysconfig` itself before importing an agent. No agent
dependency was exempted. The complete rerun passed both packages (six rounds);
there were nine technical Linux rounds total, including the preserved first
attempt. No validation games were repeated and no held-out games were played.

Raw evidence: `results/dqn/d12_final_20260921/linux_verified/` plus build and
earlier failed-check logs in its parent. The base image emits a deprecation
notice; this did not prevent the build or package execution. Floating dependency
repositories remain a rebuild limitation; the tested immutable image is locked.

## Project checks and stopping point

Full pytest: **931 passed in199.76s**. Targeted candidate/package tests:
**3 passed in12.07s**. Ruff check and format pass (166 Python files), Pyright
reports zero errors/warnings, and `git diff --check` passes. The final checker
adjustment is exercised by the completed Linux run; no agent code changed.

The three requested preparation steps are complete. Held-out evaluation, final
winner selection, submission/PR and Git publication remain unstarted. D6/D7
quality failures and the reduced-D8 limitations are unchanged.
