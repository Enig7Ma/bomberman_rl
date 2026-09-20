# D10: one controlled 3-step stage-2 run

## Protocol fixed before training (2026-09-18)

User authorized one run after the implementation/preparation task. Hypothesis:
3-step returns may improve propagation of action consequences. This is an early
D10 arm, not an established diagnosis or a completed strict D8 comparison.

Control: original D7 stage2, n_step=1, lr=3e-4, seed0, final loot mean28.4.
Treatment: **only n_step=3** plus experiment name/paths. The exact curriculum is
`dqn_d10_nstep3.json`; the wrapper asserts its equality to the baseline after
normalizing those two fields. Additional budget300,000 real transitions, finish
the last round; seed0 only. Same40/40/20 scenario mixture, encoder/mask/reward,
epsilon schedule, warm-up/cadence, target_every1000 and lr3e-4. One CPU Torch
thread. No stage3/4, other settings or budget extension.

Both start from `results/dqn/d6_20260917/pilot_lr3e-04_target1000/run_0/`.
The treatment reconstructs its returns from every original replay start using
the validated full-state fork, retaining online/target, Adam, RNG,50,047 global
transitions and11,262 updates. The preparation-only directory is retained;
the actual run has a separate directory. Parent/control files are hashed and
must remain unchanged.

Primary: final mean loot-crate coins against28.4 on the same ten seeds500–509,
with paired differences. **Improvement and passing stage2 are separate:** the
unchanged threshold is mean>43.05 and zero suicides/10 evaluated rounds.
Secondary: coin-heaven collection/steps, classic results, depth of the100k dip,
and the existing bombing/loop metrics on the same three final-model loot maps.
Final planned budgets are matched; actual round-boundary counts are reported.
Intermediate best snapshots are descriptive, not substitutes for the final.
One training seed does not establish general superiority.

Initial evaluation reuses the archived D6 results after exact equality checks of
all inference weights. New evaluation at approximately100k,200k,300k additional
transitions uses the original D7 evaluator: NumPy train=False, epsilon0, explicit
model load, no fallback, solo seat0,400-step maximum, seeds500–509 for loot,
classic and coin-heaven. Actual snapshot counts must be reported. No held-out
seeds3000–3099. Failed/unfinished rounds remain in results.

Fixed probe_v2 checksum:
`008253a6aa52af1e006b0d3858591f4a0cc94c03729073e659fb7ce010e3189f`.
Check at start, snapshots and the existing10,000-update cadence; retain existing
per-action/update Q checks. Nonfinite Q or |Q|>50 stops the run without automatic
retry. Rising loss alone is not an exclusion rule. Ordinary multi-step replay
has the off-policy bias documented in the implementation report.

```powershell
.venv/Scripts/python.exe -m docs.experiments.dqn_nstep_run --out results/dqn/d10_nstep3_seed0_20260918
```

The wrapper uses the existing driver, snapshot/evaluation and diagnostic harness.
It does not alter the agent or the learning implementation. Original D6/D7 and
the lower-lr experiment retain their results and limitations.

## Results

The single run completed on2026-09-18 while the session was inactive; its saved
results were audited on2026-09-19. **No restart, additional training or repeated
evaluation was needed.** The primary hypothesis is **not supported at the fixed
final budget**: final loot coins fell from28.4 to **1.7** (difference−26.7), with
losses on all ten paired maps. The unchanged stage2 criterion is **not passed**.
Zero suicides in the ten loot rounds satisfies only its safety component.

### Exact budget, state and timing

| Measure | Control n_step=1 | Treatment n_step=3 |
|---|---:|---:|
| Planned additional transitions | 300,000 | 300,000 |
| Actual additional transitions | 300,299 | 300,134 |
| Final global transitions | 350,346 | 350,181 |
| Additional updates | 75,075 | 75,034 |
| Final global updates | 86,337 | 86,296 |
| Stage rounds | 859 | 854 |
| Training wall seconds | 915.31 | 1,166.45 |
| Evaluation seconds | 51.76 | 37.81 |
| Stage transitions/s | 328.08 | 257.31 |

Training time includes driver, world startup during training, chunk saves and
snapshot archiving; it excludes the initial full-state fork and subtracts the
timed evaluations. It is not an isolated learner benchmark. Treatment evaluation
reuses the initial30 rounds, so its evaluation time is not directly comparable
to control's four fresh30-round evaluations. Treatment executed90 new snapshot
evaluation rounds plus3 final diagnostic rounds; those diagnostic rounds are
outside the reported training/evaluation timers. No extra seeds or budget.

Final full state:
`results/dqn/d10_nstep3_seed0_20260918/checkpoints/transition_000350181/checkpoint.pt`

SHA-256:
`4ccebac6dc3311be9cd07e385664c70b4785b0548882a7aecd61cae49ee5af64`.
Live checkpoint/replay/q_net files equal the final archive byte-for-byte. Strict
load reports a consistent full pair,100,000 replay records, write position50,181,
zero skipped updates, empty pending queue and a completed curriculum cursor.
103 protected D6/control artifacts retain their original checksums. This
checkpoint is the experimental result, **not a recommended replacement model**.

Actual scenario sampling retained the original stochastic chunk rules:

| Scenario | Rounds /854 | Round share | Transitions /300,134 | Transition share |
|---|---:|---:|---:|---:|
| Loot-crate | 320 | 37.47% | 128,000 | 42.65% |
| Classic | 344 | 40.28% | 137,600 | 45.85% |
| Coin-heaven | 190 | 22.25% | 34,534 | 11.51% |

Configured40/40/20 weights select chunks, not a forced transition quota. Shorter
coin rounds and random chunk selection explain differing realized shares; they
were not changed to match outcomes. Epsilon is0.3 at start,0.09185 after149,868
stage transitions and0.05 after180k through the end. Actual lr before the first
update was0.0003, target_every1000; no counter/schedule reset was observed.

### Learning curves at actual snapshot boundaries

Each cell is the mean over the same ten maps, including unfinished rounds.
The initial row is shared D6. Counts below are **stage/global transitions**.

| Arm | Stage/global | Loot coins | Loot crates | Loot bombs | Classic coins | Classic crates | Classic bombs | Coin-heaven steps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Shared D6 | 0 /50,047 | 20.4 | 48.5 | 42.7 | 1.2 | 16.3 | 51.0 | 124.7 |
| 1-step | 100,570 /150,617 | 2.1 | 10.5 | 4.0 | 0.9 | 20.9 | 12.2 | 124.7 |
| 3-step | 100,962 /151,009 | 33.1 | 84.4 | 57.0 | 3.6 | 48.4 | 57.2 | 124.7 |
| 1-step | 200,607 /250,654 | 26.5 | 66.7 | 47.5 | 3.2 | 47.1 | 55.8 | 124.7 |
| 3-step | 201,537 /251,584 | 23.0 | 57.0 | 50.2 | 2.3 | 27.8 | 56.4 | 127.8 |
| 1-step final | 300,299 /350,346 | 28.4 | 68.6 | 46.5 | 2.6 | 30.8 | 56.1 | 124.7 |
| 3-step final | 300,134 /350,181 | 1.7 | 9.4 | 2.9 | 3.2 | 49.7 | 27.3 | 124.7 |

All loot/classic rounds in these rows reach400 steps. Every listed evaluation
has0 suicides/10,0 deaths/10,0 invalid actions and0 timeouts. With no opponents,
raw score equals coins. Navigation collects all50 coins on10/10 maps at every
snapshot. At201,537 transitions its mean duration temporarily rises to127.8;
final duration returns to124.7. The final map step counts are
`131,120,125,121,130,117,120,125,126,132`: mean≤126, but not every map≤126.
This does not retroactively declare the original D6 criteria passed.

Around100k the control drops18.3 loot coins relative to D6; treatment instead
gains12.7. However treatment then drops10.1 and21.3 across its next evaluations.
Its **best saved intermediate** is33.1 at100,962 stage transitions, still below
43.05. The baseline's best saved stage2 result is its final28.4. Choosing the
treatment's best intermediate would answer a different question from the
predeclared final-budget comparison, and is not used to claim success.

### Final paired coins

| Map seed | Loot1 | Loot3 | Difference3−1 | Classic1 | Classic3 | Coin-heaven1/3 |
|---|---:|---:|---:|---:|---:|---:|
| 500 | 36 | 5 | −31 | 1 | 9 | 50/50 |
| 501 | 12 | 1 | −11 | 7 | 9 | 50/50 |
| 502 | 36 | 0 | −36 | 3 | 0 | 50/50 |
| 503 | 34 | 5 | −29 | 2 | 3 | 50/50 |
| 504 | 13 | 1 | −12 | 3 | 4 | 50/50 |
| 505 | 23 | 2 | −21 | 3 | 0 | 50/50 |
| 506 | 42 | 2 | −40 | 2 | 4 | 50/50 |
| 507 | 40 | 0 | −40 | 1 | 0 | 50/50 |
| 508 | 21 | 0 | −21 | 3 | 0 | 50/50 |
| 509 | 27 | 1 | −26 | 1 | 3 | 50/50 |

Across all ten final maps, loot WAIT counts are923/4,000 for control versus
**3,809/4,000 (95.23%)** for treatment. Classic WAIT is1,125/4,000 versus
2,479/4,000. These are observed actions, not inferred from a reduced score.

### Same three-map diagnostic (500–502, final snapshots)

Definitions and harness are unchanged from the control diagnostic. No hidden
coin information enters the agent. Counts pool3 rounds /1,200 actions.

| Metric | 1-step | 3-step |
|---|---:|---:|
| BOMB allowed /actions | 355/1200 | 750/1200 |
| BOMB chosen with positive yield /allowed positive-yield opportunities | 69/156 | 8/19 |
| Bombs placed | 141 | 8 |
| Bombs destroying crates /exploded bombs | 69/140 | 8/8 |
| Pending bombs at end | 1 | 0 |
| Crates /placed bomb | 197/141 =1.40 | 34/8 =4.25 |
| Empty-yield/no-attack bombs /all placed bombs | 72/141 =51.06% | 0/8 =0% |
| Empty-yield/no-attack bombs /empty-context decisions | 72/838 | 0/761 |
| WAIT /actions | 296/1200 =24.67% | 1136/1200 =94.67% |
| Immediate A→B→A returns /position triples | 13/1194 | 3/1194 |
| Decisions with reachable visible coins /actions | 361/1200 | 759/1200 |
| Reachable visible coins at the final pre-action state, per map | 0,0,0 | 2,4,0 |
| Reached400-step limit /rounds | 3/3 | 3/3 |

Treatment has fewer useless bombs and immediate returns, but this accompanies
inactivity, not an improvement in collection. On map500 it waits360 times and
collects5 coins; on501 it waits376 times and collects1; on502 it waits all400
times and collects0. The harness verifies that score/bombs/crates match the
ordinary evaluation. It does not establish why Q ranks WAIT this way; no new
reward, mask, encoder or learner modification was made in response.

### Numerical diagnostics

No NaN/Inf or |Q|>50 stop occurred. Snapshot max|Q| values are
4.211→19.806→25.947→29.469. Seven periodic probes at global updates20k–80k also
pass. The observed overall probe maximum is29.469. Last-snapshot policy churn
is43.45% **relative to the preceding probe check**, not relative to the200k
snapshot. Q threshold compliance does not certify useful decisions.

Training diagnostics below weight each round mean by its actual number of
updates. Windows assign whole rounds by their ending stage_position; they are
not claimed to be exact per-update transition bins. Zero-update records do not
contribute.

| Round-end stage range | Round records | Updates | Mean loss | Mean abs TD | Mean pre-clip gradient norm |
|---|---:|---:|---:|---:|---:|
| (0,100k] | 292 | 24,941 | 0.32872 | 0.59799 | 1.26973 |
| (100k,200k] | 279 | 25,044 | 0.19960 | 0.40489 | 1.01166 |
| (200k,end] | 283 | 25,049 | 0.17407 | 0.37762 | 0.88678 |

Loss and TD error decline while final loot behavior deteriorates. This is
evidence against judging this run from loss alone, not proof of numerical
divergence or proof that off-policy bias caused the failure. The confirmed
outcome is a worse final policy dominated by waiting on these loot maps.

### Conclusion and artifacts

**Do not adopt this 3-step setting as the working replacement.** It improves the
early saved snapshot but fails the primary final-budget comparison and stage2.
Keep n_step=1,lr=3e-4 as the existing working setting, with its own unpassed
stage2 threshold explicitly retained. Final coin collection in navigation is
preserved; classic mean improves slightly, but neither rescues the primary
result. One seed cannot establish general superiority/inferiority of n-step
learning. No further experiment, training seed or curriculum stage was started.

Compact results, per-map values, probe records, diagnostic counts and snapshot
checkpoint paths/checksums: [dqn_d10_nstep3_results.json](dqn_d10_nstep3_results.json).
Original summaries and raw logs stay in the separate results directory. Initial
evaluation was copied only after exact weight checks; every new evaluation
explicitly loaded its snapshot via the existing checked NumPy evaluator.

Code SHA: `12d74985eac3249e0b978e4fb9de5765033beabf`, plus
`dqn_nstep_run.py` and this report. Windows11, Python3.12.10, NumPy2.5.2,
Torch2.14.0, pygame2.6.1, SciPy1.18.1. Full source hashes/versions and the
experiment source copy are recorded in the run's `environment.json` and
`nstep_run_source.py`.

Pre-run verification: **915 tests passed in131.87s**, Ruff check/format (142 files),
Pyright0 errors/warnings. No implementation changed during or after training;
the full suite was not repeated for report-only analysis. Final git diff --check
passes. Changed/created files for this run are the runner, this report and the
compact results JSON. No checkpoint/replay is staged. No commit, push or merge
was performed.
