# D7 stage2: controlled LR repeat, seed0

## Preregistered before execution

The user authorized this single early tuning trial relative to the plan's order.
Historical D6/D7 results remain unchanged: D6 has no selected pilot winner and
original D7 stage2 failed (>43.05 coins required, actual28.4; suicides0/10).

Hypothesis: smaller updates mitigate the100k greedy ranking reversal and cycling
without preventing useful crate learning by the final planned budget.

Treatment changes **only learning rate3e-4→1e-4** and the experiment name/output
paths. Parsed configurations are compared programmatically. Parent is exactly
`results/dqn/d6_20260917/pilot_lr3e-04_target1000/run_0/checkpoint.pt`, SHA256
`4de4b2cfae5b0794e6ab4cc53b0c126c6d590afd057ef412013b9c47139811cd`.
Never initialize from any D7 checkpoint. Restore online/target, Adam moments,
replay/ring position, RNG and global counters (50,047 transitions,11,262 updates).
After optimizer restore, explicitly set every param_group lr=1e-4; verify all
remaining learner state equals the parent and log actual lr before first update.
Assert lr on every subsequent update, including after chunk reconstruction.

Keep seed0,300,000 additional transitions with final-round completion,
onehot_e3/best_tier, target1000, gamma.99, c_coin.5, crate/death aids0, replay100k,
update every4 transitions, original global warmup, chunk5 rounds and full chunk
saves. Keep stage epsilon.3→.05 over180k stage transitions, then.05. Mixture remains
loot/classic/coin-heaven40/40/20 by chunk selection probability. Different policies
may change round lengths, realized shares and final overrun; identical paths are
not required. Actual stage/global counts and snapshots must be reported.

Use the original D7 evaluation implementation: initial and≈100k/200k/final
snapshots,10 maps500–509 per scenario, train=False, epsilon0, NumPy inference,
explicit archived weights and no fallback. No evaluation updates or training RNG
changes. Fixed probe_v2 checksum
`008253a6aa52af1e006b0d3858591f4a0cc94c03729073e659fb7ce010e3189f`, initial/snapshots
and every10k global updates; preserve NaN/Inf/|Q|>50 stopping without retry.

**Primary:** final loot mean versus control28.4 at equal planned300k budget,
with paired differences for all10 maps. An increase is an improvement over this
control, not necessarily a stage pass. **Stage pass remains >43.05 and0 suicides.**
Navigation retention:50 coins on10/10 maps and mean steps≤124.7; report suicides,
classic results and100k depth relative to both parent20.4 and control2.1.
Compare final policies first; best intermediate values are descriptive only.

For mechanism checks, replay loot maps500–502 at≈100k and final (six diagnostic
rounds), using the existing definitions: allowed/productive bomb opportunities,
empty-context bombs, actual crate-destroying explosions, pending bombs, WAIT,
A→B→A immediate backtracks, reachable visible coins and400-step limit.
Compare final with the saved control final diagnostics; near100k with1111/1194
control backtracks and11/939 productive choices. The previously proposed strong
mechanism target is≥20.4 coins near100k and fewer backtracks. If only loss improves,
the behavioral hypothesis is unsupported. Report mixed outcomes explicitly.
One training seed cannot establish robust superiority. No other lr/seed/budget
extension/stage3/4 is included.

## Reproduction

```powershell
.venv/Scripts/python.exe -m docs.experiments.dqn_d7_lr_repeat --out results/dqn/d7_stage2_lr1e4_seed0_20260918
```

Fresh destination required; no automatic retry. Raw source/version hashes,
parent provenance, exact-state checks and actual first-update lr are saved beside
the run. Original D6 and control files are hashed before/after. Initial fork and
comparison checks precede the original training timer. The original timer excludes
evaluation, includes startup/chunk restoration/saves/probes; final diagnostic
games are separate. No trained artifacts are added to Git.

Code starts at4adb616 plus the tracked changes for this task. Original training,
reward, mask, network and replay implementations are not changed. Only the
diagnostic collector accepts optional root/count/scenario arguments to reuse the
same definitions for the new archives.

## Results

**The primary hypothesis is not supported by this seed.** Final loot-crate is
**22.9 versus28.4**, paired mean difference **−5.5 coins** (−19.37%). Stage2 also
**fails**, because22.9 is not>43.05. Suicides remain0/10 and navigation is retained.
Classic improves to4.0 from2.6; this secondary result does not replace the primary
criterion or establish superiority of this learning rate.

One run was started, then resumed after a session interruption. No new
initialization, additional training seed or budget extension was performed.
Final totals: **300,396 stage transitions /350,443 global transitions**,
**75,099 stage updates /86,361 global updates**,860 stage rounds. Control:
300,299/350,346 transitions,75,075/86,337 updates,859 rounds. Overrun396 versus299
is from completing the last round, not increasing the300k planned budget.

### Paired evaluation curves

All snapshots use ten validation seeds500–509 per scenario. The exact initial
scores, crates, bombs and navigation steps match the control baseline. All
loot/classic rounds reached400 steps. Every snapshot/scenario has deaths0/10,
suicides0/10 and timeouts0/10; invalid actions are0/4,000 crate-task actions and
0/1,247 navigation actions. No incomplete round was excluded.

| Arm | Stage transitions | Global transitions | Global updates | Loot coins | Loot crates | Loot bombs | Classic coins | Classic crates | Classic bombs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shared parent | 0 | 50,047 | 11,262 | 20.4 | 48.5 | 42.7 | 1.2 | 16.3 | 51.0 |
| control3e-4 | 100,570 | 150,617 | 36,405 | 2.1 | 10.5 | 4.0 | 0.9 | 20.9 | 12.2 |
| treatment1e-4 | 100,295 | 150,342 | 36,336 | 1.5 | 6.3 | 3.1 | 0.9 | 11.8 | 31.4 |
| control3e-4 | 200,607 | 250,654 | 61,414 | 26.5 | 66.7 | 47.5 | 3.2 | 47.1 | 55.8 |
| treatment1e-4 | 200,856 | 250,903 | 61,476 | 3.9 | 13.2 | 11.7 | 0.7 | 14.7 | 41.3 |
| control3e-4 final | 300,299 | 350,346 | 86,337 | **28.4** | 68.6 | 46.5 | **2.6** | 30.8 | 56.1 |
| treatment1e-4 final | 300,396 | 350,443 | 86,361 | **22.9** | 56.1 | 44.0 | **4.0** | 66.3 | 35.8 |

All rows retain coin-heaven50 coins/round,10/10 complete rounds,124.7 mean steps,
range117–132, and the exact original ten step counts. D6's historical per-round126
failure is not reclassified. Zero suicides on ten crate maps is not a safety
guarantee.

The100k loot decline from the shared20.4 parent is−18.9 treatment versus−18.3
control. Treatment is also22.6 coins behind control near200k. The preregistered
strong mechanism target (at least20.4 near100k) is not met. Recovery occurs by
300k but finishes below control. Among **nonfinal trained** snapshots, best loot
is control26.5 at200,607 and treatment3.9 at200,856. Including the untrained-for-
stage2 parent would give treatment's intermediate best20.4, explicitly a baseline,
not learned improvement. The final snapshot is the best recorded trained loot
snapshot for each arm. Best nonfinal classic is control3.2 at200k and treatment0.9
at100k; final treatment4.0 exceeds these. No best intermediate replaces a final.

| Validation seed | Control final loot | Treatment final loot | Difference |
|---:|---:|---:|---:|
| 500 | 36 | 32 | −4 |
| 501 | 12 | 8 | −4 |
| 502 | 36 | 35 | −1 |
| 503 | 34 | 40 | +6 |
| 504 | 13 | 13 | 0 |
| 505 | 23 | 37 | +14 |
| 506 | 42 | 12 | −30 |
| 507 | 40 | 19 | −21 |
| 508 | 21 | 1 | −20 |
| 509 | 27 | 32 | +5 |
| Mean | **28.4** | **22.9** | **−5.5** |

Three maps improve, six worsen and one ties. Complete per-map comparisons for
classic/navigation are in [dqn_d7_lr1e4_results.json](dqn_d7_lr1e4_results.json).
This is one paired training seed, not proof that1e-4 is generally worse.

### Bomb behavior and cycling on the same three loot maps

Six diagnostic rounds were run after training: maps500–502 at100,295 and300,396.
They use the previous collector with only root/count/scenario parameters added;
definitions, inference RNG and data available to the agent are unchanged.
Scores, bomb counts and crates matched the already saved evaluation for all six.
Each row below pools3 rounds/1,200 actions; all hit400 steps, with no deaths.

| Metric | Control≈100k | Treatment≈100k | Control final | Treatment final |
|---|---:|---:|---:|---:|
| BOMB allowed decisions | 1131 | 756 | 355 | 302 |
| Productive BOMB choices /allowed productive opportunities | 11/939 | 5/753 | 69/156 | 63/160 |
| Placed bombs | 11 | 7 | 141 | 150 |
| Bombs destroying≥1 crate | 11 | 5 | 69 | 63 |
| Bombs pending at limit | 0 | 0 | 1 | 1 |
| Crates/placed bomb | 39/11=3.545 | 16/7=2.286 | 197/141=1.397 | 196/150=1.307 |
| Empty-context bombs /placed bombs | 0/11 | 2/7 | 72/141=51.06% | 87/150=58.00% |
| Empty-context bombs /empty contexts | 0/227 | 2/23 | 72/838 | 87/560 |
| WAIT /1200 actions | 11 | **1162** | 296 | 154 |
| Immediate A→B→A backtracks /1194 triples | 1111 | **2** | 13 | **95** |
| Reachable visible coins before last action, summed across3 rounds | 7 | 2 | 0 | 1 |

Empty means bomb_yield=0 and attack=none; pending bombs are not counted as known
failures. “Destroying” is attributed to actual engine explosions without exposing
hidden coins to the agent. Final destroying/placed fractions are48.94% control
versus42.00% treatment. The ratio among exploded bombs is69/140 versus63/149.
Immediate backtracks omit WAIT and longer cycles. Reachability is spatial BFS on
currently free cells with bomb cells blocked, not guaranteed safe travel.

The apparent100k improvement in immediate-backtrack count is misleading:
**96.83% WAIT replaces oscillation**. Seed500 waits395 times and leaves a reachable
coin; seed501 waits367 times and leaves another; seed502 waits all400 steps.
The latter's mask does not allow BOMB, so that round is not counted as753 productive
opportunities. Reduced movement is not progress. At final, treatment has both
more empty bombs and more immediate backtracks than control. One reachable coin
remains before the last action on map502; none do in the three control final
traces. Less WAIT at final does not compensate for lower loot collection.

Across all ten final loot maps, the separately retained original evaluation
definition gives empty bombs257/440 (58.41%) treatment versus220/465 (47.31%)
control. These ten-map proportions and the three-map diagnostic proportions have
different denominators and must not be mixed.

### State, numerical checks and actual mixture

The parsed control/treatment configuration difference is exactly params.lr and
experiment name. Parent checksums match the original D7 provenance. Immediately
after loading, the full inherited learner state equals D6 except Adam param_group
lr; online/target weights, Adam moments/steps, all private RNG and replay contents
are unchanged. `first_update_lr.json` records **lr[0.0001]**, updates_before11,262.
Every update asserts1e-4, including after session resume and chunk reconstruction.
The final strict checkpoint/replay reload passes with exact_history=True,
100,000 replay entries, stage_start50,047 and stage_position300,396.

All round epsilon values match the original stage schedule. There is no second
warmup or counter reset. Mixture uses the same172 chunk selections (65 loot,
69 classic,38 coin-heaven):

| Scenario | Treatment rounds | Treatment transitions | Control rounds | Control transitions |
|---|---:|---:|---:|---:|
| loot-crate | 325 | 130,000 | 324 | 129,600 |
| classic | 345 | 138,000 | 345 | 138,000 |
| coin-heaven | 190 | 32,396 | 190 | 32,699 |

Round totals860/859 and transition totals300,396/300,299 differ because policies
produce different round lengths and stop at a completed round. This is permitted,
not a changed scenario-selection rule or a requirement for identical trajectories.

No NaN/Inf or |Q|>50 stop occurred. Snapshot max|Q|:
4.210668→11.487917→17.568945→21.805935; periodic probe checks are in compact JSON.
Final probe policy churn10.45% is relative to the preceding periodic check, not
necessarily the preceding archived snapshot. Lower final update-weighted20-round
loss0.049897 versus control0.051875 does not overcome worse behavior. Treatment
first/last20 loss0.091299→0.049897, |TD|0.334224→0.140669, gradient norm before
clipping0.404133→0.536181. No monotonic-loss acceptance filter was introduced.

### Interruption and timing limitations

The original tool process disappeared during a session interruption. Before any
continuation, no Python process was running and the strict pair was validated at
**252,903 stage /302,950 global transitions,74,488 updates**. The existing driver
cursor147 completed chunks was resumed, not manually edited. The new path refuses
a recorded diagnostic failure or already completed run. It does not recreate the
parent or duplicate initial/100k/200k evaluations. First resume checkpoint checksum
and source are retained in resume_provenance.json/resume_source.py.

```powershell
.venv/Scripts/python.exe -m docs.experiments.dqn_d7_lr_repeat --resume --out results/dqn/d7_stage2_lr1e4_seed0_20260918
```

This command records how the interruption was handled; the completed directory
now refuses another resume. No learning-rate/algorithm change was made on resume.

| Time measurement | Control | Treatment |
|---|---:|---:|
| Complete original training wall timer, excluding evaluation | 915.311s | unavailable after process interruption |
| Complete evaluation timer | 51.760s | unavailable for first3 evaluations |
| Sum of recorded chunk wall times | 796.109s | **23,990.251s** |
| Resume wall interval, including final evaluation | — | 180.438s |
| Final evaluation within resume | — | 14.060s |
| Resume interval excluding evaluation | — | 166.378s |

Two chunk records contain long execution gaps: index135=768.152s and
index136=22,684.830s,2,000 transitions each. The exact cause of these gaps was not
instrumented, so they are not relabelled as measured compute time. The remaining
170 chunks total**537.268s for296,396 transitions**, or551.672 transitions/s.
This is an explicitly filtered descriptive rate, not the full experiment time or
a speed advantage attributable to lr. Control's corresponding unfiltered chunk
rate is377.208 transitions/s; chunk timers omit the subsequent full-save hook,
unlike the complete original training timer. Gaps, interruption, machine load and
measurement boundaries prevent a fair speed claim. No missing time was invented
or silently discarded from the raw results. Six post-training diagnostic games
are separate from these timing denominators.

### Artifacts and conclusion

Raw root: `results/dqn/d7_stage2_lr1e4_seed0_20260918/`. Each compact result row
links to its archive/checksum. The final full-state result is
`checkpoints/transition_000350443/checkpoint.pt`, SHA256
`0686968193dcb200c38fad2a8e81a824e5451862ca33c3e75277a17f10d58ab7`, with adjacent
replay.npz and q_net.npz. Run-root files carry the same final state.
Code SHA, executed source hashes, inherited-state proof, first-update lr, resume
provenance, all original checksums, exact budgets and per-map comparisons are
stored in [dqn_d7_lr1e4_results.json](dqn_d7_lr1e4_results.json) and raw manifests.
Original D6 and D7 checksums passed the final verification.

**Do not promote1e-4 from this comparison:** primary final loot is worse, the100k
failure is deeper and largely changes form from movement cycles to waiting, and
the final useless-bomb fraction rises. Navigation and numeric checks pass and
classic improves, so this is a mixed result, not numeric divergence. Stage2 still
fails. One seed is insufficient for a general claim about learning rate. No other
configuration, seed, longer training, stage3/4, commit, push or merge was run.

### Validation and changed files

Related tests5 passed, including parameterized full-state transfer with lr3e-4
and1e-4, retaining Adam moments and the requested lr after another driver resume.
Final mandatory checks: **891 tests passed in163.48s**, Ruff check passed,
Ruff format134 files passed, Pyright0 errors/warnings, git diff --check passed.
The full suite was repeated after adding the interruption-resume wrapper, not
after every training snapshot. Final strict reload additionally verified all
three live files equal their final archive and103 protected D6/control files
retain their original hashes.

Changed: docs/experiments/dqn_d7_diagnose.py (optional collector selection),
tests/test_training_continuation.py (learning-rate preservation regression).
Created: docs/experiments/dqn_d7_lr_repeat.py, dqn_d7_lr1e4.json,
dqn_d7_lr1e4.md and dqn_d7_lr1e4_results.json. All original experiment reports,
result summaries and training/agent implementation files remain unchanged.
