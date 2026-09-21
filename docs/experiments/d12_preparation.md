# D12 preparation: candidate inventory and standalone packages

**Subsequent completion, 2026-09-21:** blended validation, candidate/protocol
freeze and Docker/Linux verification are now recorded in
[d12_freeze_and_linux.md](d12_freeze_and_linux.md). Pending items and proposed
protocol below describe the earlier preparation state; use the frozen protocol
and runtime lock linked from that report for the current status.

2026-09-21, `feature/ivan-dqn`, base `b4a67a2a6685fc483d53019ecd496c3a86cfc27c`.
This completes packaging preparation, not full D12 or a submission decision.
D6/D7 quality gates remain failed; reduced D8 is complete within its recorded
scope. No new training, stage3/4, tuning or held-out3000–3099 games were run.

Plan: `C:/Users/ivans/Downloads/Telegram Desktop/dqn.md`, D12. No AGENTS.md or
CLAUDE.md found in the repository/ancestors. The user confirmed that dqn.md and
the repository are all available material; a separate tabular Q12 is unavailable.
The proposed final protocol below therefore names its assumptions explicitly.

Machine-readable evidence: [d12_manifest.json](d12_manifest.json).
Raw checks, ZIPs and local weight backups: `results/dqn/d12_preparation_20260921_v2/`.

## Candidates, without silent replacement

| Identity | Recorded training transitions | Role |
|---|---:|---|
| Baseline DQN stage2 seed0 | 350346,86337 updates | provisional practical DQN package |
| Maria blended table, commit40b5718 | 6586885 in metadata,18400 rounds | provisional practical table package |
| Previous deployed table, commit507662e | 6586885 in metadata | historical control, backup retained |
| D8 table seeds0/1/2 | 350146/350150/350687 | independent matched-study models, retained separately |
| D8 DQN seeds0/1/2 | 350187/350272/350310 | independent matched-study models, retained separately |

All nine NumPy files were strictly loaded and their schema, metadata and hashes
recorded. DQN baseline SHA256 is
`fa724eb4ebebefd2170cb15afe18289e79d023743916252c5ba4238d687c7f33`.
Its existing full checkpoint/replay remain in
`results/dqn/d7_stage2_seed0_20260918/checkpoints/transition_000350346/`;
the previous manifest records consistent generations and Adam lr3e-4.
Missing n_step in its old metadata means1; onehot_e3, best_tier, gamma0.99,
target_every1000, c_coin0.5, crate/death aids0. No lr1e-4 or3-step model is used.

Blended table SHA256 is
`ffd992c6e5c8236d1c012d31bf43b7ed737542cdc19ab8965702c1cd3afeef2d`.
Metadata explicitly records `blended_with_heuristic: {delta: 0.05}`. Compared
with the old backed-up table,54144 Q entries differ (max absolute change
0.05000019);50596 visit entries also differ. Sum(n) changed from6586885 to6637481,
while metadata steps_trained remains6586885. Do not interpret the new visit sum
as additional observed training transitions. A complete blending script and
curriculum/seed history are not available from this commit's changed files.
The recorded config is E3/best_tier/symmetry, gamma0.9, coin potential0.5,
bomb_aid0.1 and spot_potential0.2. It is a different reward/history from DQN.

The old practical table's4.60 score does **not** describe the blended table.
The new table has only a technical smoke here, not comparable strength validation.
No D8 winner or final submission winner is declared. Do not pool the baseline
and D8 seeds as repeated trainings of one identical configuration.

## Package and loading checks

ZIPs contain the relative-import closure of callbacks plus the selected NumPy
model, at `dqn_agent/` or `tabular_q_agent/`. They have one callbacks.py at the
top, no nested callbacks, train.py, replay, checkpoints, logs or requirements.txt.
Tabular transitions/metrics remain in its closure because callbacks imports
them; they do not execute training in train=False. DQN's Torch learner is absent.
Byte-stable archive creation is tested. Original repository models are untouched.

| Package | Bytes | Files | ZIP SHA256 |
|---|---:|---:|---|
| dqn_stage2_baseline.zip | 113094 | 13 | ca9ad459781803bc13c95ee75b540d162f6aa6e978787e8be0bc4d5d0433d6e6 |
| tabular_blended.zip | 159347 | 18 | 1e1df5cc3cef352abc27d7142484ea4de3cec12ae21517ee9684b710d4e5174c |

Each ZIP was extracted into a temporary copy of the unchanged framework with
assets and random_agent, without tournament/, training/, dev/ or sibling learned
agents. Fresh `python -I` processes checked default and explicit loading, setup
and act, exact loaded arrays, and absence of Torch/foreign agent imports.
Non-stdlib imports are NumPy and measured framework dependencies; events is a
framework module and Windows colorama is allowed only through framework imports.
Default/explicit loading also ran with an unrelated cwd.

Explicit missing, corrupt and schema-incompatible files are rejected. Missing
or corrupt default models trigger the existing fallback; these deliberate tests
modify only disposable extracted copies, then restore their original bytes.
Successful package/game checks require the real model and do not accept fallback.
Source, backup, extracted model and ZIP checksums are checked independently.

## Original-framework smoke and latency

The unchanged main.py ran classic, each candidate vs3 random_agent, train=False,
three rounds, world RNG seeded500 once, one sequential process. These are a
single world's next three maps, not separately seeded500/501/502 games.
Stock opponent randomness was not altered or described as fully controlled.
This is a packaging/timing smoke, not a fair quality comparison or held-out result.

| Candidate | Setup default ms | Setup explicit ms | Timed acts | Mean ms | p99 ms | Max ms | Timeouts / skips |
|---|---:|---:|---:|---:|---:|---:|---|
| DQN baseline | 2.782 | 2.956 | 1200 | 0.427 | 0.807 | 1.026 | 0 / 0 |
| Blended table | 47.941 | 46.105 | 922 | 0.492 | 1.011 | 1.543 | 0 / 0 |

Setup includes model loading; interpreter/import startup is separate in raw
results. Act timings include all measured decisions with no warm-up discarded.
Both meet setup<2s, p99<50ms and max<250ms in this short Windows smoke. Logs
confirm no engine timeout replacements/skips; no latency guarantee across all
final tactical lineups is claimed. Source model bytes remain unchanged.
The first harness attempt failed before any game because the copied framework
needed a logs directory. The harness creates it now; failed diagnostics are
preserved under the first output directory. No completed evaluation was discarded.

## Docker status

Docker Desktop4.40.0 / Engine28.0.4, Linux amd64, is accessible when the tool has
permission to access its named pipe. Earlier sandbox checks returned missing
pipe/access errors; this is not evidence that Docker is still unavailable.
No compatible framework image is installed. The original repository Dockerfile
build and container smoke are **not performed** in this packaging-preparation
step. They remain required for D12; Windows success is not Linux/Docker evidence.
Use a minimal build context containing the unchanged Dockerfile/framework,
assets, selected extracted package and standard opponents, excluding results/,
checkpoints and .git. The full dependency-heavy image must be built before
claiming the original Docker requirement passed.

## Concrete remaining protocol, pending candidate freeze

1. Inventory/validation decision: retain baseline stage2 DQN as the provisional
   practical model. Evaluate the new blended table on the existing25 validation
   maps500–524 vs3 rules with identical positions/private opponent streams;
   reuse existing compatible DQN/reference results. Freeze identities/hashes
   before any held-out use. If a D8 model is considered, report the change
   explicitly rather than selecting a best seed silently.
2. With Q12 absent, proposed primary held-out task is repository
   `vs-rule-based`, seeds3000–3099, all four candidate seats, paired rule control.
   Secondary supplied-task presets: coin-heaven-solo, crates-solo, vs-peaceful,
   vs-coin-collector and mixed on the same seed range. This is an explicit
   operational proposal, not a quotation of a nonexistent Q12 document.
3. Report the two practical candidates separately from all three D8 seeds per
   learner; do not manufacture independent baseline seeds. Primary paired delta
   against rule control, credited kills, coins, suicides, survival, invalid and
   timeouts; cluster maps/seats appropriately, with training seed as outer unit
   for the matched study. Retain all failed and truncated games.
4. Head-to-head: frozen DQN/table plus2 rules,50 held-out seeds, all seat
   permutations as D8 specifies. Freeze policy/opponent RNG rules and schedule
   in a manifest before launch; world seed alone is insufficient.
5. Submission rule stays the plan's rule: greater held-out paired delta; if
   delta CIs overlap, fewer suicides; remaining tie goes to the table. No tuning
   after held-out results. Docker/framework smoke and jobs1 latency must cover
   the exact final package; then PR/review and post-merge verification.

No step above was launched automatically in this preparation task. The next
bounded action is the25-map blended-table validation and concrete freeze manifest,
not additional training. Missing Q12 no longer blocks inventory/packaging;
the final schedule must simply identify the chosen operational assumptions.

## Reproduction and checks

```powershell
# Uses a fresh output directory; creates two ZIPs and runs six technical rounds.
.venv/Scripts/python.exe -m docs.experiments.d12_prepare --out results/dqn/d12_preparation_repeat
# Rebuild compact evidence from existing checks without games:
.venv/Scripts/python.exe -m docs.experiments.d12_prepare --summarize results/dqn/d12_preparation_20260921_v2 --out docs/experiments/d12_manifest.json
.venv/Scripts/python.exe -m pytest tests/test_dqn_agent_package.py -q
```

Full pytest: **931 passed in140.10s**. Ruff check/format161 files, Pyright0
errors/warnings and git diff --check pass. The offline exporter was run on the
saved evidence after its addition; no training or smoke repetition was needed.
No Ruff/Pyright exclusions were added. ZIPs, models and raw logs remain ignored.
