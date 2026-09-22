# Practical candidates, D8 gaps and D12 preparation

**Latest, 2026-09-21:** the three authorized preparation steps are complete:
25-map blended-table validation (score5.84), candidate/protocol freeze, and full
Docker build with successful Linux checks for both packages. See
[report](d12_freeze_and_linux.md), [frozen protocol](d12_frozen_protocol.json)
and [runtime lock](d12_runtime_lock.json). No held-out games, final winner,
submission or Git publication yet. The earlier pending statements below record
the state before this work and do not supersede this update.

**D12 packaging preparation, 2026-09-21:** [manifest](d12_manifest.json) and
[checks/proposed final protocol](d12_preparation.md) now cover nine model
identities and isolated ZIP checks for baseline DQN and Maria's blended table.
Full D12, Docker build, blended-table strength validation and held-out evaluation
remain pending. The user confirms no separate tabular Q12 document is available.

**2026-09-21 update:** the user subsequently authorized and completed reduced D8
(both learners, three fresh seeds, navigation50k + crates300k). The transition
driver prerequisite described below is now implemented. See
[conclusions](d8_reduced_summary.md), [protocol](d8_reduced_protocol.md) and
[results](d8_reduced_results.md). The dated audit below remains historical;
its "not executed" statements refer to2026-09-19. D6/D7 remain failed, full
D8 all-seat head-to-head/combat curriculum is still omitted, and D12 is pending.
The study did not replace candidate weights. A subsequent upstream integration
brought Maria's blended table; see the summary's publication note. It has not
been evaluated here and does not inherit the old table's validation score.
The new matched study is separate from the old models' practical validation.

2026-09-19. Branch `feature/ivan-dqn`; initial tree clean, HEAD
`af6b6f3d0cae2d9002639c9410dbdcce6bf562c9`. No training in this task.
Plan read from `C:/Users/ivans/Downloads/Telegram Desktop/dqn.md`, D8/D12.
No repository/parent AGENTS.md or CLAUDE.md found. No stage4, new hyperparameters,
dense encoder, CNN or held-out games are authorized or executed.

## Experiment status remains unchanged

D6 did not pass its original criteria; no pilot winner was selected. Stage2
baseline loot28.4 did not pass >43.05. The lr1e-4 repeat (22.9), n_step3 repeat
(1.7), and hunting continuation (loot0.7, coin-heaven1.9; combat score0.20/0.68)
remain negative results. Hunting lost navigation. None is retrospectively
classified as successful. Reports and original checkpoints are retained.

## Candidate manifest and provenance

Machine-readable inventory: [dqn_candidates_manifest.json](dqn_candidates_manifest.json).
All selected weights were strictly loaded, not inferred from filename/mtime.

| Candidate | Evidence | Use |
|---|---|---|
| DQN baseline stage2 `results/dqn/d7_stage2_seed0_20260918/checkpoints/transition_000350346/` | checkpoint/replay generations agree; 350346 transitions,86337 updates; D6 seed0 full-state parent recorded in continuation.json | practical finalist |
| `agent_code/tabular_q_agent/model/q_table.npz` | introduced by Maria Druz commit507662ec402c059af2c05bcfd6aff67e5cfabf11, Train agent; metadata18400 rounds,6586885 steps and nonempty visits | practical finalist |
| `results/dqn/d5_throughput_20260917/tabular_q_agent/run_0/` | D5 benchmark run/config/logs;870/1558/2758-step snapshots and2758-step final | smoke artifacts, not quality finalists |

The manifest contains hashes, full available metadata and all five discovered
tabular files (including the deployed model). No additional trained table was
found in the inspected local results/agent_code/training/dev trees. The deployed
table's full training curriculum and independent training-seed history are not
available; saved seed100007 and stage nav-refresh are final config values only.
Do not describe them as complete provenance of all6.59M steps.

DQN: n_step missing in the old format explicitly means1; actual Adam groups
lr3e-4, target_every1000, onehot_e3, best_tier, gamma0.99, c_coin0.5,
crate_aid=death_aid=0. Replay100000, consistent, not stale. Parent metadata and
all three SHA-256 values are in the manifest. Full parent source was read only.

Inference backups created and hash-verified, outside Git:

- `results/dqn/candidate_backups_20260919/dqn_stage2_q_net.npz`;
- `results/dqn/candidate_backups_20260919/tabular_q_table.npz`.

DQN full training state remains in the source archive (checkpoint/replay/model).
The table's tracked model is also recoverable from its Git commit; no separate
optimizer state applies to it. These local backups are not off-machine disaster
recovery: copy the directory and checksum manifest to team storage before handoff.
Other DQN arms remain in their original result directories and are not finalists.

## Practical versus strict D8 compatibility

| Condition | DQN baseline | Deployed table | Matched? |
|---|---|---|---|
| State abstraction | E3, onehot_e3 | discrete E3 index | yes; shared feature/mask/core/symmetry files byte-identical |
| Mask/canonicalization | best_tier, canonical | best_tier, symmetry=true | yes |
| Base score | coin + credited kill | same | yes |
| Discount | 0.99 | 0.9 | no |
| Coin potential | 0.5 | 0.5 | coefficient yes, shaping differs with gamma |
| Other reward | crate/death aids0, no bomb/spot aids | bomb_aid0.1, spot_potential0.2, crate/death0 | no |
| Curriculum | documented D6 navigation then stage2 mix | full history unavailable; final nav-refresh | unverified/different |
| Training seeds | one seed0 | final saved seed100007, complete history unavailable | no |
| Budget | 350346 transitions | 6586885 steps | no |
| Validation here | classic vs3 rules, seeds500–524, seat0 | identical | yes |

This is a **practical model comparison**, not completed strict D8 or an isolated
test of function approximation. Existing solo control results on500–509 can be
reused as practical retention evidence: table47.1 vs DQN28.4 loot coins; classic
solo8.9 vs2.6. See `dqn_controls.md`. Stage3 evaluations use different opponents;
the D2 random-initialized network is a different model. Neither supplies the new
required combat evaluation. No compatible25-map result with this opponent-stream
protocol was found, so75 new rounds were played. No existing evaluation was
rerun simply to improve a result.

## Validation protocol fixed before collection

Classic, max400 engine steps, candidate seat0 vs three rule_based_agent;
reference is four rule_based_agent with seat0 measured. Seeds500–524, jobs1.
Framework seeded starting-position permutation is retained: same seed, same
lineup seat, same board/coins/positions across arms, verified by first-observation
hashes. This is not all-seat evaluation.

Learned candidates: train=False, no trainer, epsilon0, explicit absolute model
path, exact loaded-array assertions, private policy RNG seed500 each round.
Rule-based uses independent NumPy/Python streams `10*world_seed + seat`, including
reference seat0; opponents1–3 receive the same streams across arms. Stock setup
entropy seeding is isolated and caller RNG restored. Strategies are unchanged.
Fixed streams do not imply identical trajectories or remove the original
training-opponent randomness; world seed alone is not full RNG control.

The evaluator runs the world to its normal end, retaining posthumous credited
kills. No hidden information is fed to a policy. Before/after hashes confirm
weights and parent checkpoint/replay unchanged. Results and full per-map values:
[dqn_candidates_validation.json](dqn_candidates_validation.json).

| Candidate | Mean score | Delta to reference [95% interval] | Kills/round | Coins/round | Suicides/25 | Survival/25 | Invalid total | Timeouts |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| DQN stage2 | 3.20 | +0.44 [-0.80,1.64] | 0.24 | 2.00 | 1 | 23 | 11 | 0 |
| Table | 4.60 | +1.84 [0.80,2.96] | 0.28 | 3.20 | 1 | 20 | 22 | 0 |
| Rule-based reference | 2.76 | 0, reference | 0.04 | 2.56 | 18 | 6 | 149 | 0 |

Intervals: paired nonparametric bootstrap of25 game seeds,20000 draws,
private NumPy seed20260919, percentile95%. They **do not represent variability
of independent trainings**. Table-minus-DQN mean1.40, paired interval[0.00,2.76];
do not claim a resolved robust advantage. The table leads this practical sample;
DQN's improvement over reference is not resolved. Both learned samples have
suicide rate1/25=0.04, not a safety guarantee. D12 selection is still pending.

75 rounds took93.425s including setup, I/O and strict weight checks. An initial
attempt failed callback signature validation before any round; functools.wraps
fixed the harness. No completed game was repeated or discarded. Raw logs:
`results/dqn/candidates_validation_20260919_run/`.

```powershell
# Reproduction only: requires a NEW output directory; do not repeat for this task.
.venv/Scripts/python.exe -m docs.experiments.dqn_candidates --out results/dqn/candidates_validation_repeat
```

## Concrete D8 completion plan (not executed)

Full D8 requires matched inputs/mask/reward/curriculum/training seeds/budgets,
curves every100k transitions and at least3 independent seeds for each learner;
raw strength, sample/time efficiency, reaching safe-random+1, visit-count bins
0/1–9/10+, worst-seed spread, and head-to-head on50 validation seeds with all seat
permutations. None of these is completed by the practical table above.

**Implementation prerequisite:** training/config.py currently rejects transitions
for table stages; training/driver.py plans tabular chunks and epsilon in rounds.
Merely swapping `agent` in a DQN curriculum does not work. Add and test an
opt-in common transition-budget/snapshot/epsilon schedule for the table while
preserving legacy round configs. Finish rounds and record actual overshoot;
ensure reward equivalence, no teacher data, independent RNG and full continuation.
This is proposed work, not a change performed here.

Under the current prohibition on stage4, full-plan D8 cannot be completed.
The actionable reduced option is **navigation50k + crates300k**, seeds0/1/2,
both learners, E3/best_tier/canonical, gamma0.99, coin potential0.5, all other
aids0, teacher_share0; common scenario mixture and stage-relative epsilon,
common validation maps/seat schedule. Keep DQN lr3e-4,target_every1000,n_step1;
table alpha defaults remain its learner-specific update. This compares complete
learners, not only a neural versus tabular representation.

Use fresh runs for clean reproducibility after the scheduling adapter; optionally
reuse the existing DQN seed0 only after proving exact protocol compatibility,
including task draws, feature version and epsilon schedule. The D6 archive has
additional seeds but no matching complete stage2 arms. Do not silently combine
them with a changed curriculum. Existing6.59M-step table cannot be a matched arm.

Reduced evaluation: initial model and actual~100k/200k/300k/final boundaries,
same25 rule-based maps plus10 navigation/loot maps; a safe-random reference for
the curve threshold; visit-bin diagnostics from the matched table; report
training-seed spread. This still omits hunting/combat curriculum and full D8
50-seed/all-seat head-to-head unless separately authorized. Label it reduced D8,
not full D-S8 acceptance. No new learning is started by this document.

Timing uses measured D5 end-to-end118.93 DQN and227.51 table transitions/s,
including startup/warm-up/saves, excluding evaluation. Recent hunting189.54 DQN
transitions/s shows task dependence; use D5 conservatively, not plan's1M/hour.

| Scope | DQN serial time | Table serial time | Combined training only |
|---|---:|---:|---:|
| Reduced350k x3 seeds each | 2.45h | 1.28h | 3.73h |
| Full1.75M x3 seeds each (NOT authorized) | 12.26h | 6.41h | 18.67h |

Allow roughly0.5–1h reduced evaluation/analysis overhead, plus scheduling-adapter
implementation/tests (planning allowance2–4h, not a measured benchmark). Thus
reduced completion needs approximately6–9h of work/serial execution; no promise
of speedup from jobs2 until contention is measured. Full D8 also needs substantial
head-to-head time:50 maps x24 seat permutations=1200 games (~25min at this sample's
1.25s/game, before other evaluations). Opponent mix and duration can change it.
These estimates are not authorization to launch full training.

## D12 / submission checklist

- Freeze candidate identities, config and checksums before held-out use. Preserve
  all failed experiments; agree reduced-D8 scope and remaining acceptance gaps.
- Create separate staging directories/zips for each candidate; do not replace
  the shipped table or silently install the DQN model. Currently
  `agent_code/dqn_agent/model/q_net.npz` is absent. Include selected NumPy model
  under model/, callbacks at agent root, its self-contained modules only;
  exclude training checkpoints/replay/logs and nested callbacks.
- Fresh-process NumPy-only import/setup/act tests for BOTH staged candidates,
  explicit/default/missing/corrupt loading, schema/action order; no Torch or
  training/tournament/sibling agent dependencies on inference path. Account for
  legitimate framework Windows dependencies. Existing network/callback/package
  tests are useful but do not certify a newly produced zip.
- Measure staged setup including load<2s, jobs1 decision p99<50ms,max<250ms,
  zero timeouts. Current games recorded latency but this instrumented run is
  not a dedicated D12 latency certification. Previous D2 numbers were for an
  untrained model, not a substitute.
- Test extracted packages with the original framework and Docker. Existing
  Dockerfile installs a large training stack including Torch; successful Docker
  inference alone would not prove NumPy-only packaging. No Docker build was
  performed in this task; document image/framework version and results later.
- Only AFTER freeze: joint Q12/D12 seeds3000–3099, same maps/seats and all available
  training seeds; required lineups and50 held-out head-to-head maps. Agree missing
  tabular Q12 details before launch. Keep validation separate. No held-out seed
  was used here.
- Preserve the plan's preregistered submission rule: larger paired held-out delta
  to rule-based; overlapping delta CIs -> fewer suicides; remaining tie -> table.
  Do not choose a new rule after seeing held-out scores. Single-seed reliability
  remains an explicit limitation unless additional matched seeds are authorized.
- Prepare a scoped PR with manifest/results, exclusions and checks; review with
  Maria, then rerun required tests/package smoke after merge and verify the zip
  model checksum. No PR publication, commit, push or merge in this task.

## Verification

Directed RNG test passed: reproducible streams, matching opponent seats between
reference/candidate arms, distinct streams per seat, no global RNG mutation.
Ruff check/format148 files passed; Pyright0 errors/warnings. Full pytest:
**920 passed in98.37s**. `git diff --check` and explicit whitespace checks of
the new untracked files passed. Validation integrity:75 completed games,
25 matching initial-board/position signatures, no held-out seeds, source weights
unchanged. No training or model installation was performed.
