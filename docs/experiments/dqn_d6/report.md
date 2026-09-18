# D6 navigation: pilot and seed replication

Experiment date: 2026-09-17. Baseline commit
`1739492d4135823114b9e83abd7088e3540d88d4`, branch `feature/ivan-dqn`.
The tree was clean before D6. Local plan:
`C:/Users/ivans/Downloads/Telegram Desktop/dqn.md`, D6, sections 5.8 and 8.
No project/ancestor AGENTS.md or CLAUDE.md was found. No D7, reward/feature
changes, commit, push or merge. Production agent/driver code is unchanged.

The experiment code and preregistration are tracked beside this report.
Each run records `environment.json` with exact source SHA256 hashes, versions,
and Git status, plus a copy of its executed `experiment_source.py`.
Python 3.12.10; NumPy 2.5.2; Torch 2.14.0; pygame 2.6.1; scipy 1.18.1.

## Protocol and interpretations

Coin-heaven solo, onehot_e3, best_tier, 50,000 transitions per training seed,
default gamma/rewards/warm-up 5,000/batch 64/train_every 4. Epsilon .3 to .05 over
30,000 transitions. Five-round chunks; snapshots at the chunk boundary crossing
10,000,20,000,... and the final boundary. Actual counts appear below and in
`learning_curves.csv`; no snapshot is relabelled as an exact threshold.

Four pilot combinations use independent fresh **identical seed 0 initialisations**
and identical conditions, followed by fresh seeds 1/2 initialisations only for
replication. Maximum two concurrent training processes, one Torch CPU thread
per process. No smoke weights or prior-seed checkpoint is used for initialisation.

The preregistered eligibility filter requires finite bounded Q, no abort,
completed budget, and last 20 updated-round mean loss <= 1.25 times first 20
updated-round mean loss + 1e-6. Among eligible configurations, ranking uses
final validation mean coins, full-clear fraction, then all-round mean steps.
Ties use (3e-4,1000), (3e-4,250), (1e-4,1000), (1e-4,250). Loss is not a ranking
score; no intermediate checkpoint is cherry-picked.

**Step criterion:** the plan says 50 coins every round and <=126 steps without
explicitly saying whether the step cutoff is a mean. This report preregistered
the conservative interpretation: each of ten validation rounds must clear all
50 coins within 126 engine steps. The mean-based interpretation is also reported
separately, never substituted for the strict criterion after seeing results.

Evaluation includes the untrained network and every snapshot on seeds 500..509,
coin-heaven-solo, with_control=False, train=False, explicit policy=learned,
epsilon effectively zero and NumPy QNetwork. Instrumented setup verifies the
actual file and every loaded weight array, so fallback/current-model substitution
fails loudly. Checkpoint, replay and live NumPy model checksums are compared
before/after evaluation; the checkpoint includes the persisted training RNG.
All rounds remain in denominators. `mean_steps_success` is null if none clear;
`mean_steps_all` includes unsuccessful rounds and their full engine duration.

## Fixed probe

`results/dqn/d6_20260917/probe/probe.npz`: 2,000 observed states, 1,000 sampled
without replacement from each source. bfs_agent supplied 1,144 source observations;
rule_based_agent 1,243; both played coin-heaven solo on seeds 900..909 only.
Actor/extractor/subsampling RNG seeds and row provenance are in the raw manifest.
Rule-based global Python/NumPy RNGs are explicitly seeded after its entropy-seeding
setup; global RNG state is restored after collection. No learner is constructed.

- schema_id: `3963eb07bf146d11`
- file SHA256: `8cbef87bc7b7c9853fe1d6872756c1e31953ef4180c4fc71c94aa834a5ad05b5`
- content fingerprint: `f3f0d67563978b8779786ca807f24aaaebfe5588d82d0d761a5482de79f66379`
- **Only 6 unique canonical vectors.** Repeated observations are intentionally
  retained. This is a narrow expert-navigation probe, not broad state coverage
  for crates/combat. It cannot establish bounds on unseen states. Existing D4
  guards additionally check states encountered by the learner.

Two independent test collections produced byte-identical files. The same immutable
probe is used everywhere. Initial archived replay is empty; final replay counts
match the sum of actual executed training actions, with no probe insertions.
Initial/snapshot probe checks supplement the existing every 10,000-update checks.
A threshold failure stops that specific run and preserves its traceback/weights
when serialisable; a regression test exercises this path with deliberately bad Q.

## Four-combination pilot (training seed 0)

| lr | target updates | Actual transitions | Updates | Time s | Transitions/s | Loss first 20 -> last 20 | Max probe absQ | Final coins | Clear fraction | Mean / max steps | Eligible |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---|
| 3e-4 | 1000 | 50047 | 11262 | 89.61 | 558.52 | .02125 -> .08643 | 4.211 | 50 | 10/10 | 124.7 / 132 | no |
| 3e-4 | 250 | 50095 | 11274 | 88.00 | 569.25 | .02343 -> .12350 | 13.670 | 50 | 10/10 | 124.7 / 132 | no |
| 1e-4 | 1000 | 50010 | 11253 | 134.95 | 370.58 | .02038 -> .08530 | 4.411 | 50 | 10/10 | 124.7 / 132 | no |
| 1e-4 | 250 | 50068 | 11268 | 136.07 | 367.95 | .02326 -> .14042 | 12.961 | 50 | 10/10 | 124.7 / 132 | no |

Times are measured around training: world setup, full saves, snapshot archives,
probe measurements included; top-level Python/Torch imports, version capture and
validation excluded. Two processes run concurrently. These are actual process
rates under changing host load, not a controlled speed ranking of hyperparameters.
No timeout/death/invalid action was seen in pilot validation (including initial
models). Initial models average 0.2 coins and run to 400 steps: those failures
remain visible in all curves.

**Selection failed:** no configuration passes the preregistered growth filter.
`selection.json` records selected=null. No successful winner is declared and
no rule was relaxed after observing results. This limited pilot is not evidence
that one setting is superior. The operational loss filter is stricter than the
plan's qualitative wording; failure of this filter alone does not prove numerical
divergence. Q remains finite and far below 50; bootstrap target scale changes
with training, so the loss-growth mechanism needs separate investigation.

## Learning curves

Each cell is actual transitions / mean coins / mean steps (all ten rounds).
At every trained snapshot all ten rounds collected 50 coins; therefore successful-
round and all-round mean steps coincide. Initial models have no complete rounds.

| Combination | Initial | Snapshot1 | Snapshot2 | Snapshot3 | Snapshot4 | Final |
|---|---|---|---|---|---|---|
| 3e-4 /1000 | 0 /.2 /400 | 10103 /50 /124.7 | 20344 /50 /124.7 | 30124 /50 /124.7 | 40221 /50 /124.7 | 50047 /50 /124.7 |
| 3e-4 /250 | 0 /.2 /400 | 10044 /50 /124.7 | 20699 /50 /127.8 | 30241 /50 /124.7 | 40872 /50 /124.7 | 50095 /50 /124.7 |
| 1e-4 /1000 | 0 /.2 /400 | 10372 /50 /127.8 | 20692 /50 /124.7 | 30626 /50 /124.7 | 40255 /50 /124.7 | 50010 /50 /124.7 |
| 1e-4 /250 | 0 /.2 /400 | 10259 /50 /127.8 | 20410 /50 /127.8 | 30123 /50 /127.8 | 40598 /50 /124.7 | 50068 /50 /124.7 |

The learned navigation appears by the first snapshot and then largely plateaus.
For the default pilot, 99.55% of probe decisions change between initial and first
snapshot; subsequent snapshot policy churn is 0. Growing loss/Q thus coexists
with a stable navigation policy on this narrow probe. No D7 overestimation
experiment was performed, so the cause is not claimed as confirmed.

Every CSV curve row includes its inference snapshot and full archived checkpoint
path and SHA256. Each archive also retains the matching replay and probe.json.
Raw data live only in `results/dqn/d6_20260917/`.

## Checks

Five directed D6 tests passed. Full pytest: **882 passed in 97.47s**, no skips.
Ruff check passed; Ruff format --check: 123 files formatted; Pyright 0 errors/
warnings; git diff --check passed. No agent/driver production changes or checker
exclusions. The final verification includes the result-cache guard; the suite was
not repeated for each unchanged-code training run.


## Diagnostic replication on three seeds (not an accepted selected setting)

Since every pilot failed eligibility, **no configuration was selected**. The
original default 3e-4 / target_every=1000 was then replicated on fresh seeds 1
and 2 to carry out the requested three-seed diagnostic. This is a transparent
post-pilot decision, recorded as an amendment in protocol.md; it does not
change selection.json or turn the rejected baseline into a tuning winner.
The existing pilot seed 0 is reused, not rerun.

| Seed | Source | Actual transitions | Rounds | Updates | Time s | Transitions/s | Loss first 20 -> last 20 | Max probe absQ | Final coins / clears | Mean / max steps |
|---:|---|---:|---:|---:|---:|---:|---|---:|---|---|
| 0 | reused pilot | 50047 | 319 | 11262 | 89.61 | 558.52 | .02125 -> .08643 | 4.211 | 50 / 10 of 10 | 124.7 / 132 |
| 1 | diagnostic replication | 50115 | 322 | 11279 | 102.76 | 487.70 | .02168 -> .08874 | 4.368 | 50 / 10 of 10 | 124.7 / 132 |
| 2 | diagnostic replication | 50072 | 317 | 11269 | 102.80 | 487.10 | .04477 -> .08815 | 4.591 | 50 / 10 of 10 | 124.7 / 132 |

Seed 1 snapshots: 0, 10114, 20250, 30083, 40268, 50115 transitions.
Seed 2 snapshots: 0, 10444, 20507, 30504, 40292, 50072 transitions.
Both trained curves reach 50 coins / 124.7 mean steps at the first snapshot and
remain there. Seed 1 starts at .2 coins / 400 steps with no clears. Seed 2's
untrained model is different: 49.9 coins on average, 9/10 full clears, but all
rounds reach 400 steps. These results are retained; a favourable initial coin
score is not a trained-model result.

The reported step statistic is engine round duration, including any delay from
outstanding bombs/explosions; it is an upper bound on first collecting coin 50.
Every final evaluation has **zero bombs**, so round completion coincides with
clearing all coins and the final threshold comparisons are unaffected by this
distinction. Initial/intermediate round-duration curves do not claim separate
first-50-coin timestamps. All six final models have zero validation deaths,
invalid actions and timeouts. Trained rounds all clear, so the successful-round
mean and all-round mean are both 124.7. Initial failed rounds are never excluded
from the all-round mean.

Initial weight-array SHA256 checks show all four pilots use identical seed-0
weights, while seeds 0/1/2 have three distinct initialisations. See
initialisation_checks.json. All six runs have exactly the same probe checksum;
initial replay length is zero, and final replay length equals actual training
actions. No probe states or smoke training transitions enter replay.

Across the six unique runs: **300,407 transitions, 1,881 training rounds,
654.19 summed process-seconds of training**. The process seconds must not be
read as elapsed wall time because runs were paired. There are 36 evaluated
checkpoints (including six initial networks), each with ten validation rounds.
No seed or failed evaluation was dropped. This is a limited pilot/diagnostic,
not a claim of generalisation to unseen seeds or tasks.

## D6 acceptance result

| Criterion | Result |
|---|---|
| All 50 coins in every final evaluation, three baseline seeds | **Pass: 30/30** |
| <=126 steps in every round (preregistered strict reading) | **Fail: max 132 on each seed** |
| <=126 steps as a mean, shown separately | Pass: 124.7 on each seed |
| Probe Q finite and absQ <=50 | Pass: max 4.591 for the three-seed baseline; max 13.670 across all pilots |
| Loss not growing under the preregistered operational filter | **Fail on all four pilots and all three baseline seeds** |
| Stable configuration selected and confirmed | **Not achieved: selection remains null** |

**D6 acceptance is not achieved. D7 is not started.** The common navigation
behaviour is reproducible, but that does not override the failed loss-growth
filter or strict step cutoff. Follow-up should first examine how the changing
bootstrap targets affect the loss diagnostic and resolve the intended step
aggregation with the project owners. Neither threshold is silently changed here.

## Reproduction and artifact map

Use the repo's virtualenv, from the repository root. On a fresh checkout,
collect the fixed probe once; subsequent runs reuse that same file:

```powershell
.venv/Scripts/python.exe -m docs.experiments.dqn_d6 collect --out results/dqn/d6_20260917/probe
```

The four exact pilot commands (execute the first pair concurrently, wait for
both, then execute the second pair concurrently; maximum two processes):

```powershell
.venv/Scripts/python.exe -m docs.experiments.dqn_d6 run --config docs/experiments/dqn_d6/lr3e-04_target1000.json --out results/dqn/d6_20260917/pilot_lr3e-04_target1000/run_0 --seed 0
.venv/Scripts/python.exe -m docs.experiments.dqn_d6 run --config docs/experiments/dqn_d6/lr3e-04_target250.json --out results/dqn/d6_20260917/pilot_lr3e-04_target250/run_0 --seed 0
.venv/Scripts/python.exe -m docs.experiments.dqn_d6 run --config docs/experiments/dqn_d6/lr1e-04_target1000.json --out results/dqn/d6_20260917/pilot_lr1e-04_target1000/run_0 --seed 0
.venv/Scripts/python.exe -m docs.experiments.dqn_d6 run --config docs/experiments/dqn_d6/lr1e-04_target250.json --out results/dqn/d6_20260917/pilot_lr1e-04_target250/run_0 --seed 0
.venv/Scripts/python.exe -m docs.experiments.dqn_d6 select --out results/dqn/d6_20260917/selection.json --pilots results/dqn/d6_20260917/pilot_lr3e-04_target1000/run_0 results/dqn/d6_20260917/pilot_lr3e-04_target250/run_0 results/dqn/d6_20260917/pilot_lr1e-04_target1000/run_0 results/dqn/d6_20260917/pilot_lr1e-04_target250/run_0
```

The two explicitly diagnostic baseline replications, also run concurrently:

```powershell
.venv/Scripts/python.exe -m docs.experiments.dqn_d6 run --config docs/experiments/dqn_d6/lr3e-04_target1000.json --out results/dqn/d6_20260917/diagnostic_baseline/run_1 --seed 1
.venv/Scripts/python.exe -m docs.experiments.dqn_d6 run --config docs/experiments/dqn_d6/lr3e-04_target1000.json --out results/dqn/d6_20260917/diagnostic_baseline/run_2 --seed 2
```

The run command evaluates initial and all snapshots automatically. To repeat
only evaluation, preserving all training artifacts:

```powershell
.venv/Scripts/python.exe -m docs.experiments.dqn_d6 evaluate --out results/dqn/d6_20260917/diagnostic_baseline/run_1
```

The final script returns a cached completed result on repeat `run`, preserving
original training timing; another seed/config in that directory is refused.
This result-preservation guard was added after the training runs and tested;
it does not affect their learning or evaluation. Each run's archived source
records the exact version that produced it. New independent trainings require
new output directories; do not present a no-op resume as a throughput test.

Tracked compact tables: pilot_summary.csv, replication_summary.csv,
learning_curves.csv and training_diagnostics.csv. The latter reports
update-weighted loss/TD/gradient means per snapshot interval; raw per-round
metrics retain the full curves. Each learning-curve row includes snapshot and
full checkpoint paths and checksums; each archived checkpoint directory also
contains its matching replay. Compact probe provenance is probe_manifest.json;
full 2,000-row selection provenance is in results/. All models and replay files
are ignored, not final trained submissions.


## Integration before publication

Before committing D6, feature/ivan-dqn was fast-forwarded from 1739492 to
8c9af52 (including merge 3eac8cc and tabular commit 28db8c4). Incoming changes
replace PEP 695 annotations with TypeVar/Generic in both copies of bookkeeping
and symmetry, ignore .DS_Store, and update the submitted tabular archive.
The archive's only changed file content is tabular_q_agent/symmetry.py;
three directory entries were added. No training algorithm change was found.
Local D6 files were preserved and restored without conflicts. Experiment results
above still refer to their original baseline and archived source hashes;
training was not repeated merely to integrate annotation compatibility changes.

Post-pull verification: **882 tests passed in 85.97 s**; Ruff check/format,
Pyright (0 errors/warnings) and git diff --check passed. The user subsequently
authorized committing and pushing D6; only experiment scripts, configurations,
tests and compact reports are included, not models/replay/raw results.


## Post-hoc audit, 2026-09-18

See [audit/report.md](audit/report.md) for criterion provenance, exact loss
windows, target-sync associations, original probe reconstruction, and probe_v2
results on all 36 existing snapshots. Original selection remains null; no
training or D7 was started. The original probe/results/checksums are retained.
