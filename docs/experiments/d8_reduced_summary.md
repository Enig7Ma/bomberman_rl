# Reduced D8: conclusions and completion status

2026-09-21. The authorized reduced D8 is complete: two learners, three fresh
training seeds each, navigation50k + crates300k,30 evaluated models and1395 new
validation rounds. The25 compatible rule-reference games were reused. No
held-out seeds, hunting/stage4, new hyperparameters or model installation.

Protocol, commands and checks: [d8_reduced_protocol.md](d8_reduced_protocol.md).
Generated curves/tables: [d8_reduced_results.md](d8_reduced_results.md).
Per-map results, checkpoint paths/hashes, environment and diagnostics:
[d8_reduced_results.json](d8_reduced_results.json).

## Final results at the same planned budget

Each combat mean uses25 maps500–524; each solo mean uses10 maps500–509.
Group means weight the three training seeds equally. These are final models,
not selected best intermediate models.

| Learner/seed | Actual transitions | Combat score | Loot coins | Loot suicides/10 | Navigation coins | All50/10 | Navigation steps, all rounds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Table0 | 350146 | 2.16 | 19.8 | 0 | 50.0 | 10 | 400.0 |
| Table1 | 350150 | 2.16 | 12.0 | 0 | 50.0 | 10 | 127.8 |
| Table2 | 350687 | 1.12 | 9.4 | 0 | 32.9 | 2 | 348.2 |
| DQN0 | 350187 | 2.84 | 5.7 | 0 | 3.9 | 0 | 400.0 |
| DQN1 | 350272 | 2.84 | 3.7 | 0 | 50.0 | 10 | 124.7 |
| DQN2 | 350310 | 4.88 | 29.0 | 0 | 50.0 | 10 | 124.7 |
| Table mean | — | 1.813 | 13.733 | 0/30 | 44.300 | 22/30 | — |
| DQN mean | — | 3.520 | 12.800 | 0/30 | 34.633 | 20/30 | — |
| Shared safe-random | no training | 1.08 | 11.5 | 0/10 | 23.3 | 0/10 | — |

Rule-based reference combat mean=2.76. DQN-minus-table final differences by
training seed are+0.68,+0.68,+3.76; paired map95% intervals are[-0.36,1.68],
[-0.56,1.96],[2.36,5.24]. The hierarchical training-seed/game interval for the
mean difference+1.707 is[0.213,3.640]. This supports an advantage in this reduced
protocol, with a substantial contribution from seed2; three training seeds do
not establish robust general superiority. DQN seeds0/1 have only+0.08 delta to
rule reference (map intervals include zero); seed2 has+2.12[0.60,3.72].
All three final tables are below reference. Final combat suicides: DQN2/75,
table1/75; this is not a safety guarantee. No evaluation timeouts occurred.

All six loot results fail the unchanged >43.05-coin threshold, despite0/10
suicides each. Navigation retention is inconsistent. The <=126 criterion is
applied to mean engine round length across all10 scheduled maps, together with
all50 in every map, as in prior reports: only final DQN1/2 meet both. Table0
collects50 but averages56.1 bombs and reaches400 in every navigation round.
The engine requires no remaining bombs/explosions to finish a one-survivor round
(`environment.py`, time_to_stop); recorded round length is not time of the50th
coin. That latter metric was not collected and is not substituted retrospectively.
Table2's successful-only mean141.0 excludes eight failures; its all-round348.2
is the primary number. No failed round is dropped.

## What the curves show

All learners reach50 navigation coins after the navigation stage, then crates
training can degrade that skill. DQN0 loses it at the final snapshot; table0/1
temporarily lose it and recover, while table2 ends with partial retention.
Loot performance is non-monotonic and weak: DQN0/1 finish below their earlier
loot peaks, while DQN2 recovers to29.0 after lows4.4/4.9. No common quality
improvement appears across all seeds. Loss was not used to choose models.

At sampled checkpoints, DQN seeds0/1 cross safe-random+1 combat score at50047/
50115 transitions. DQN2 already exceeds that threshold at initialization
(2.56 combat,49.9 navigation coins), so its reported crossing0 is not learning
efficiency evidence. Tables cross at250276/250448/150486. Crossings are neither
interpolated nor guarantees of staying above threshold: final table2 falls below.

Visit diagnostics include all three tasks and show DQN differences from the
table on visited states. On unvisited states all table Q values tie, so100%
membership in its greedy set is uninformative. Different policies visit
different observations; score attribution cannot isolate a generalization cause.
This compares full learners: table per-visit max-Q updates versus replayed
Double-DQN/Adam, as well as representation, not function approximation alone.

The40/40/20 curriculum probabilities apply to chunk selection, not a quota of
transitions. Actual stage2 coin-heaven transition shares were18.0/12.7/14.9%
for tables and10.4/9.0/11.4% for DQN. The exact counts by scenario and rounds
are in the generated table. Shorter successful navigation rounds explain why
transition shares can be below selection probabilities. No mixture was changed.

## Integrity, timing and artifacts

All archived models/checkpoints/replays still match their60 saved file hashes.
Evaluation explicitly loaded each model, checked exact arrays and matching
initial board/position hashes, disabled training/exploration, and preserved
source files. Decision/score attribution totals equal engine outcomes. Probe_v2
checksum is unchanged; maximum checked abs Q over all runs=22.4745, below50.
Passing the numeric guard did not prevent poor behavior.

Table training elapsed357.1–374.5s (935–982 transitions/s). DQN0/1 took936.8/
889.7s (374/394 transitions/s). Times include warm-up, saves, archive/probe and
worker contention; they exclude evaluation and process import startup. DQN2
took26021.0s: chunks123/124 account for5181.75/19572.48s. The cause of these
long delays cannot be established from saved logs. Full elapsed is retained;
its13.46 transitions/s is not a representative throughput benchmark, and no
pause duration was guessed or removed. Study wall time including evaluation,
startup and those delays=28501.50s (7h55m01.5s).

The table2 Windows derived-metrics replacement failure and exact continuation
are preserved in [d8_recovery.json](d8_recovery.json) and the protocol amendment.
The original cached failed study summary remains in results; final aggregation
uses the recovered run. Its missing final evaluation took77.07s, partly alongside
the original evaluator; only13.16s was added to total elapsed to avoid double
counting. No interrupted run was replaced by a fresh seed.

Root: `results/dqn/d8_reduced_20260921/`. Each learner's `run_N/archive/` stores
the exact checkpoints named in the JSON. Final transition suffixes are the six
counts in the table above. Table archives contain q_table.npz; DQN archives
contain q_net.npz, checkpoint.pt and replay.npz. No large artifacts enter Git.
Old D6/D7/lr/n-step/hunting results and candidate manifest hashes are unchanged.

## Position in the plan and next bounded task

D0–D5 implemented; D6/D7 quality gates remain failed. The two early D10 changes
(lr1e-4 and3-step returns) remain negative. Reduced D8 is now completed, full
D8 acceptance is not: hunting/combat training and50-map/all-seat head-to-head
remain outside its authorized scope. D9/dense and D11/CNN were not started.

Next is D12 candidate freeze and packaging, followed by its separately arranged
held-out protocol and Docker/framework/latency checks. Do not silently replace
the practical stage2 DQN or deployed table with a new D8 model: validation
strength, retention, training budget and provenance differ. Agree the candidate
manifest first and retain all three matched training seeds in reporting.
No further training, held-out games or packaging was started here.

Checks:929 tests passed; Ruff check/format156 files, Pyright0 errors/warnings,
git diff --check. No exclusions or learning-rule changes.

## Upstream integration before publication

The first push was rejected because origin already contained Maria's
`40b5718` (blended deployed table/zip and a new bfs-control preset), merged
upstream as `ec96c4d`. The two unpublished D8 commits were rebased onto that
existing history without creating a merge or forcing a push. None of the three
evaluation presets used by D8 changed. All study runs explicitly use their own
fresh or archived models, so this integration does not change the experiment.

The deployed table is now the upstream blended model, SHA256
`ffd992c6e5c8236d1c012d31bf43b7ed737542cdc19ab8965702c1cd3afeef2d`.
It was not evaluated here and must not inherit the old candidate's4.60 score.
The earlier statement about unchanged candidate hashes describes the study
before this integration. The old practical table remains in its recorded backup
and Git history; its manifest/validation remain historical. D12 must inventory
this newly available blended candidate explicitly before deciding what to freeze.

Post-rebase verification: **929 passed in132.61s**; Ruff check/format158 files,
Pyright0 errors/warnings and git diff --check pass. The original practical table
backup checksum still matches its historical manifest. All six final study
checkpoints are distinct across their training seeds; DQN initial models are
also distinct. No experiment was rerun after integration.
