# Reduced D8 preregistration — 2026-09-21

User authorized the entire reduced study, then commit/push. No stage3/4,
new settings, dense encoder, CNN or held-out seeds. D6/D7 remain failed; the
existing candidates and all negative experiment artifacts remain unchanged.

## Fixed training protocol

Both complete learners, fresh seeds0/1/2,50,000 navigation +300,000 crates
transitions each, configurations `d8_reduced_{dqn_agent,tabular_q_agent}.json`.
No warm start from an old experiment. E3 abstraction, canonicalization,
best_tier mask, gamma0.99, coin shaping0.5, other aids0, no teacher.
DQN onehot_e3, n_step1,lr3e-4,target_every1000. Table uses its existing
per-visit learning-rate schedule; comparing complete learners, not isolating
function approximation. Chunk5, stage epsilon0.3->0.05 over60% of the budget.
Crates chunk probabilities40/40/20: loot/classic/coin-heaven solo.
No opponents in training; world RNG is seeded. Task draws follow the same
chunk-index algorithm; different episode lengths can produce different counts
and realized task shares. Record both, not a promise of identical trajectories.

At most two Windows spawned worker processes, one Torch CPU thread per DQN
worker. Independent directories, RNG, models/replay/logs. Finish the current
round at a budget threshold; record actual counts. Initial snapshot, navigation
endpoint, and stage2 thresholds100k/200k/300k at completed chunk boundaries.
These correspond nominally to global0/50k/150k/250k/350k, NOT exact100k global
grid points. Record actual transition and wall-time axes for every model.

Same probe_v2 SHA256
`008253a6aa52af1e006b0d3858591f4a0cc94c03729073e659fb7ce010e3189f`.
Probe at initial and archived snapshots for both learners; DQN retains its
every10k updates and per-update finite-Q checks. NaN/Inf or |Q|>50 stops that
run; preserve last committed state/failing archive and reason, do not retry or
replace the seed. Loss monotonicity is not a quality or stop criterion.

## Fixed evaluation / reporting

Every saved model: coin-heaven solo and loot-crate solo on500–509, classic
vs3 rule_based_agent on500–524, candidate seat0, seeded framework starting
positions, max400 engine steps. No seat rotation. Private candidate RNG500,
train=False, epsilon0, explicit model and exact loaded-array checks. Models and
training RNG never updated by evaluation. Rule streams `10*world_seed+seat`,
isolated NumPy/Python RNG, same as the existing candidate validation. World
seed alone is not asserted to control unwrapped opponents.

Reuse the25 matching rule-based reference games from
`dqn_candidates_validation.json`; verify initial board/position signatures.
Evaluate one shared masked safe-random policy on the same45 maps; RNG500.
Its explicit initial DQN model is loaded; random is intentional, not fallback.

Report all seeds, actual budgets/updates/timing/failure status; learning curves
by transitions and elapsed training+save/probe time. Evaluation time is separate.
Combat score and paired delta to reference, coins/kills/suicides/survival,
invalid/timeouts; solo coins, all50 fraction, all-round versus successful steps.
Primary comparison is the final snapshot at the planned budget, not each seed's
best intermediate. Report mean/min/max across3 training seeds and worst seed.
Map-bootstrap95% CIs use20,000 paired draws with RNG20260921; label these as
map variability, not training variability. Do not infer robust superiority from
three seeds alone. Threshold safe-random+1 means the first evaluated snapshot
with mean combat score >= the fixed random reference mean+1; no interpolation.

Visit diagnostics use the matched table seed/milestone (retain its actual count).
Capture the acting policy's actual single feature extraction; canonical state
visit count is sum of all6 table action counts, binned0/1–9/10+. Count selected
action membership in the table's allowed greedy set (ties accepted), decisions,
and following raw score increments, including terminal/posthumous residual.
Different policies visit different states: these observational bins do not
establish causal generalization benefits. Smoke checks their sums against score.

## Scope and commands

This is reduced D8: hunting/combat curriculum and50-map/all-seat head-to-head
are omitted, as are D12 packaging/held-out decisions. Do not claim full D-S8.
No changes to reward/mask/encoder/algorithm to improve outcomes mid-study.

```powershell
.venv/Scripts/python.exe -m docs.experiments.d8_reduced --smoke --jobs 2 --out results/dqn/d8_reduced_smoke_20260921
.venv/Scripts/python.exe -m docs.experiments.d8_reduced --jobs 2 --out results/dqn/d8_reduced_20260921
```

Smoke is isolated seed0,401 transitions per stage, chunk1, DQN warmup8 so updates
are exercised; two validation maps per preset. None of its weights/replay are
used for the study. Verify driver resume/idempotence via regression tests and
smoke completed-run checks. Actual raw versions/configs/source copy and hashes
are retained with results; no model or replay files are added to Git.

## Pre-run checks completed

Smoke completed in25.03s at jobs2. Table:1442 transitions/updates; DQN:1236
transitions,308 updates. Each produced initial and two stage-end archives and
three evaluated checkpoints. The two learners' initial board/position hashes
match on all smoke maps. Visit-bin decision/score totals match engine outcomes.
Completed-run resume returned no new chunks for both agents. The smoke source
is preserved verbatim as `experiment_source.py.txt`; a subsequent explicit
Features conversion fixes its static cross-package type, preserving values.

Full project suite before study:927 passed in135.35s. Ruff check/format and
Pyright pass after final report-only typing/formatting corrections. The offline
report generator was exercised on smoke without replaying any games. Study
source/config copies were frozen before its six training runs.

## Technical recovery amendment (after collection began)

Table seed2 encountered Windows PermissionError while replacing derived metrics,
after the Q-table and its driver cursor were already committed. The underlying
lock owner is unknown. This was not a Q-threshold stop. The committed state had
323086 transitions; the last published chunk file still reported321977.

The driver now retries only PermissionError on derived-file replacement, at most
five attempts (0.05/0.10/0.15/0.20 seconds between attempts). Permanent failure
still propagates and preserves the previous file. Directed tests cover both.
No reward, policy, schedule, RNG or budget changed. Seed2 resumed its committed
table, visits and RNG, finishing at350687;27601 further transitions were played.
The committed record prefix is identical, round IDs remain unique, visit totals
and metric step totals equal the final counter. Evidence: [d8_recovery.json](d8_recovery.json).

Original failure, committed weights, interrupted summary and exact recovery
source copies remain under `tabular_q_agent/run_2/recovery/`. The original
orchestrator had cached the interrupted summary, so final aggregation refreshes
that run from its completed summary and evaluates only its missing final model;
already collected evaluations are reused. The original study summary is retained.
This procedural exception is reported, not hidden as an uninterrupted execution.

Actual recovery command (do not repeat on a completed run):

```powershell
.venv/Scripts/python.exe -c "from docs.experiments.d8_reduced import train_worker; train_worker('results/dqn/d8_reduced_20260921', 'tabular_q_agent', 2, False, resume=True)"
```

Offline report reconstruction, without training or evaluation:

```powershell
.venv/Scripts/python.exe -m docs.experiments.d8_report --root results/dqn/d8_reduced_20260921 --out docs/experiments
```

In addition to the preregistered map intervals and seed spread, the report adds
the plan's hierarchical bootstrap (training seeds outside, games inside),20000
draws with seed20260921. This is descriptive analysis; no model-selection or
acceptance rule changes.

## Final implementation checks

After the recovery fix and report bootstrap test: **929 passed in233.50s**.
Ruff check passes, Ruff format checks156 files, Pyright reports0 errors/warnings,
and git diff --check passes. Tests ran alongside the later evaluation workers;
evaluation timings are instrumented wall times, not isolated latency benchmarks.
No exclusions were added. Infrastructure commit: `9038d04` (original local
commit02bfea6, rebased before publication).

The training source was based on `bcf35a633d225619629510ea52a4ff82ae3195bc`
plus the archived working changes, not on that Git SHA alone. Exact source hashes
and configs are in the raw environment manifest and compact results JSON.
Environment: Windows11, Python3.12.10, NumPy2.5.2, Torch2.14.0, Pygame2.6.1,
SciPy1.18.1; pytest9.1.1, Ruff0.16.7, Pyright1.1.414.
