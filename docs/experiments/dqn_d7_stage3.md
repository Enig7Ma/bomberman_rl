# D7 stage3: one diagnostic hunting continuation

## Decision and protocol fixed before training, 2026-09-19

**Stage2 did not pass its quality criterion and is not reclassified as successful.**
After the unsuccessful lr1e-4 and3-step arms, the user authorized a single
diagnostic transition to hunting despite that failed gate. This is an explicit
deviation from the intended quality-gated progression in D7, not completion of
D7 or evidence of stability across training seeds. No stage4 is authorized.

Plan inspected: `C:/Users/ivans/Downloads/Telegram Desktop/dqn.md`, D7 and its
exploration defaults. No separate tabular Q8 plan/curriculum was found in the
repository, ignored dev tree or available Downloads Markdown/JSON documents.
AGENTS.md/CLAUDE.md were not present. Use the user-specified fallback allocation.

Parent full checkpoint/replay/model (read-only):
`results/dqn/d7_stage2_seed0_20260918/checkpoints/transition_000350346/`.
Actual metadata:350,346 transitions,86,337 updates; n_step absent in v1 means1;
gamma0.99,lr3e-4,target_every1000,onehot_e3,best_tier,c_coin0.5,crate/death aids0.
The actual restored Adam param_groups must have lr3e-4. Full learner, Adam,
feature/action/replay RNG and replay are compared with the parent during fork.
Replay v1→v2 loading explicitly preserves each one-step return. No n-step3 or
lower-lr checkpoint, replay clearing or fresh initialization is used.

One training seed0,400,000 additional **learner** transitions, two consecutive
200,000-transition blocks, finishing the last round of each. Chunk size5.

| Block | Current task (80%) | Repeated earlier tasks (20%) |
|---|---|---|
| Hunting peaceful | classic vs3 peaceful_agent | loot solo8%, classic solo8%, coin-heaven solo4% |
| Hunting coin collector | classic vs3 coin_collector_agent | loot solo8%, classic solo8%, coin-heaven solo4% |

Weights select chunks stochastically; they do not promise exact round/transition
shares. The early-task mix is the previous stage2 40/40/20 mix within its20%
allocation. The second block keeps these same solo repetitions; it does not add
an unrequested peaceful replay allocation. Actual shares will be reported.

Epsilon nominally0.3→0.05 over the first240,000 hunting transitions, then0.05.
Using the existing two-block driver, block1 linearly reaches0.0916667 over200k;
block2 starts at that same value and reaches0.05 over40k. No exploration jump at
the lineup change. Finishing block1's last round adds at most399 transitions at
its endpoint epsilon; block2's schedule starts at the actual boundary, so the
global240k point can shift by that small reported overshoot. Internal driver
stage IDs2 and3 identify the two **hunting blocks**, not D7 stage4 combat.
Global counters/warm-up/updates/target cadence are preserved. Reward, encoder,
mask, lr, target_every and n_step1 remain fixed.

## Evaluation and diagnostics

Parent and snapshots near each100k additional transitions: classic vs3 peaceful
and vs3 coin-collector, exactly25 validation maps500–524, seat0 throughout,
400-step maximum, NumPy inference, train=False, epsilon0. Same layouts/seat
policy for every arm; no seat rotation, no held-out3000–3099. Snapshots use actual
chunk boundaries. Endpoint retention: coin-heaven solo and loot solo on500–509.
Parent solo results already exist with identical maps and inference settings and
are reused; learned parent combat and masked safe-random combat are evaluated
once each because no matching evaluations with this opponent-RNG protocol exist.

Candidate RNG is private seed500 every evaluation round. Evaluation wraps the
existing opponents with independent NumPy/Python streams seeded
`10*world_seed + opponent_index` (index0–2). Original setup/act rules are unchanged;
the caller's global RNG is restored after each call, including entropy-seeding
setup. Jobs1/sequential only. Fixed draws do not imply identical trajectories
after the candidate policy diverges. Training uses the stock opponents and their
entropy seeding: world seed alone does **not** fully determine opponent behavior.
This limitation prevents claiming bitwise reproducibility of the whole training
run or robust superiority from its one seed.

Raw score, engine-credited kills, coins, suicides, survival, crates, bombs,
invalids and timeouts are recorded. Forced-step and WAIT fractions pool decisions
(numerator count /total candidate decisions), not unweighted round fractions.
Empty-yield/no-attack BOMB choices use the unchanged feature definitions. The
evaluation trace observes the actual single extraction, without extra RNG draws
or access to hidden information. Realized shaped-return gaps include terminal
engine score, including posthumous credited kills. No learner update in evaluation.

Targets: peaceful mean score>21.20 **and** kills≥2.69; collector mean score>2.62.
Compare parent and safe-random explicitly, not only these published reference
numbers. One training seed /25 maps with fixed seat are diagnostic evidence,
not proof of reliable superiority. Original stage2>43.05 threshold stays failed
unless the new measured endpoint itself clears it; prior results stay unchanged.

Fixed probe_v2 SHA-256:
`008253a6aa52af1e006b0d3858591f4a0cc94c03729073e659fb7ce010e3189f`.
Check at start, snapshots and every10k global updates, plus existing per-action/
update checks. NaN/Inf or |Q|>50 stops the single run; no automatic retry or new
settings. Loss is recorded but not used alone to select a model or stop training.

## Reproduction

```powershell
.venv/Scripts/python.exe -m docs.experiments.dqn_d7_hunting --out results/dqn/d7_stage3_seed0_20260919
```

The script uses the existing full-state fork, driver and D7 snapshot/stop harness.
It writes to a new directory and will not overwrite the parent. Original
checkpoints remain protected by hashes. Configuration: `dqn_d7_stage3.json`.

## Completed results

The single run completed without a Q-stop. No retry, tuning, other training seed or stage4 was run. Stage2 remains failed.

### Learning curve

| Additional transitions | Global transitions | Global updates | Peaceful score / kills | Collector score / kills | Probe max abs Q |
|---:|---:|---:|---:|---:|---:|
| 0 | 350346 | 86337 | 2.88 / 0.16 | 2.60 / 0.16 | 21.570 |
| 100000 | 450346 | 111337 | 0.00 / 0.00 | 0.80 / 0.08 | 22.168 |
| 200000 | 550346 | 136337 | 5.40 / 0.44 | 1.88 / 0.04 | 23.886 |
| 300838 | 651184 | 161547 | 0.00 / 0.00 | 0.36 / 0.04 | 18.796 |
| 400372 | 750718 | 186430 | 0.20 / 0.00 | 0.68 / 0.04 | 11.130 |

Scores and kills are means across the same 25 maps. Safe-random: peaceful **4.52 / 0.40**, collector **1.16 / 0.08**. Final score changes versus parent are -2.68 and -1.92; versus safe-random -4.32 and -0.48. The temporary peaceful peak at200k (5.40 /0.44) is not the final result and is far below the plan references.

### Endpoint and control metrics

| Policy / lineup | Coins | Crates | Bombs | Suicides /25 | Survival /25 | Invalid total | Forced % | WAIT % | Empty-context bombs / all bombs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Parent vs-peaceful | 2.08 | 24.36 | 54.12 | 0 | 25 | 0 | 25.52 | 31.13 | 1127/1353 |
| Parent vs-coin-collector | 1.80 | 18.12 | 51.72 | 0 | 25 | 1 | 17.77 | 33.29 | 1072/1293 |
| Safe-random vs-peaceful | 2.52 | 48.16 | 39.20 | 0 | 25 | 0 | 11.55 | 28.46 | 519/980 |
| Safe-random vs-coin-collector | 0.76 | 15.80 | 37.68 | 0 | 25 | 2 | 8.01 | 27.88 | 731/942 |
| Final vs-peaceful | 0.20 | 3.92 | 7.04 | 0 | 25 | 0 | 2.34 | 3.59 | 134/176 |
| Final vs-coin-collector | 0.48 | 7.36 | 38.52 | 0 | 25 | 3 | 7.56 | 8.09 | 854/963 |

Coins, crates and bombs are per-round means; all rows above have10,000 decisions and400 mean steps. Forced/WAIT denominators are actual decisions. Empty-context means bomb_yield=0 and attack=none at selection, not proof that the explosion destroyed nothing. Timeouts are zero. Final suicides0/25 per lineup do not guarantee safety: the300838 snapshot had2/25 suicides and21/25 survival against collectors. All per-map values and intermediate metrics are in `dqn_d7_stage3_results.json`.

### Retention

| Scenario | Parent coins | Final coins | Parent steps | Final steps | Parent / final all50 |
|---|---:|---:|---:|---:|---:|
| coin-heaven solo | 50.0 | 1.9 | 124.7 | 400 | 10/10 / 0/10 |
| loot-crate solo | 28.4 | 0.7 | 400 | 400 | 0/10 / 0/10 |

Final coin-heaven per-map coins (seeds500?509): **0,0,1,3,1,1,5,6,2,0**. All20 final solo rounds survived with zero suicides/invalids/timeouts. Navigation was lost, not preserved. Coin-heaven has zero WAIT and zero bombs over4,000 actions; its failure cannot be explained by waiting alone. Loot final:3.6 crates and1.5 bombs per round, WAIT9.7%.

### Actual allocation and schedule

| Block | Task | Rounds | Transitions | Round share % | Transition share % |
|---|---|---:|---:|---:|---:|
| hunting-peaceful_agent | classic peaceful_agent,peaceful_agent,peaceful_agent | 370 | 148000 | 74.00 | 74.00 |
| hunting-peaceful_agent | loot-crate solo | 60 | 24000 | 12.00 | 12.00 |
| hunting-peaceful_agent | classic solo | 65 | 26000 | 13.00 | 13.00 |
| hunting-peaceful_agent | coin-heaven solo | 5 | 2000 | 1.00 | 1.00 |
| hunting-coin_collector_agent | classic coin_collector_agent,coin_collector_agent,coin_collector_agent | 404 | 158522 | 79.37 | 79.11 |
| hunting-coin_collector_agent | classic solo | 60 | 24000 | 11.79 | 11.98 |
| hunting-coin_collector_agent | loot-crate solo | 40 | 16000 | 7.86 | 7.99 |
| hunting-coin_collector_agent | coin-heaven solo | 5 | 1850 | 0.98 | 0.92 |

Weighted chunk sampling produced only10 coin-heaven rounds and3,850 transitions (0.962% of the whole run), below the nominal4%. This is an observed sampling imbalance and a plausible contributor to forgetting, not an isolated causal finding. The configuration was not changed mid-run.

Block budgets:200,000 +200,372; overshoot372 from finishing the final round. Epsilon started0.3; at global550346 it was0.0916667, next round0.09125, and by590533 it reached0.05; final0.05. The second-block position200372 is distinct from the whole hunting count400372.

### Persistence, time and diagnostics

Full final checkpoint: `results/dqn/d7_stage3_seed0_20260919/checkpoints/transition_000750718/checkpoint.pt`.
SHA-256: `704faf6021b7c65902f6e300e67fc9b3cbc8f6ed6c82296b1994807a0dc25dc5`.

Post-run weights_only load and replay consistency validation passed:100,000 replay rows, no stale generation, exact_history=true, empty pending queue, zero skipped updates, Adam lr3e-4, n_step1, target_every1000. Live checkpoint/replay/model hashes match the final archive. Parent checkpoint/replay/model hashes still match the preregistered source; the run also checked all17 parent files. Agent/driver source hashes remain unchanged. Each evaluation explicitly verified the requested NumPy weights; no fallback or training updates were used.

400,372 additional transitions and100,093 updates; global750,718 /186,430. 1,009 training rounds,202 chunks. Training/driver/save time2,112.339s (35.21min), evaluation621.156s (10.35min), combined2,733.494s (45.56min), excluding initial full-state fork. Throughput189.540 transitions/s,47.385 updates/s; training time includes chunk startup and persistence but excludes evaluation. Evaluation comprises250 learned combat rounds,50 random control rounds and20 final retention rounds;20 parent solo results were reused.

All10 periodic probe measurements and5 snapshot measurements were finite and below50; maximum24.2962 at global130,000 updates. Final maximum11.1301. No failure file or Q-stop. Lower Q/loss is not evidence of a good policy. First/last round means: loss0.054246 /0.014954, absolute TD0.164379 /0.060525, gradient norm0.483584 /0.079678. These are individual round summaries, not a monotonicity claim. Full curves remain in metrics.jsonl.

WAIT shows strong intermediate degeneration against peaceful:99.98% at100k and95.76% at300838; final3.59%. Thus the final poor result is not simply the same WAIT-dominated policy. Final empty-context bomb fractions are134/176 (76.14%) peaceful and854/963 (88.68%) collector. The evidence shows unstable useful behavior and loss of earlier skills; it does not isolate learner, representation, exploration or task-mixture causality.

### Criteria and limitations

- Peaceful score>21.20: **failed (0.20)**; kills>=2.69: **failed (0)**.
- Collector score>2.62: **failed (0.68)**.
- Final collector has one credited kill in25 rounds; peaceful has none. Temporary11 kills/25 at200k do not establish a retained hunting skill.
- Earlier navigation and crate collection were not retained. Stage2 remains failed; its final loot0.7 is also below43.05.
- Probe safety passed, but does not establish policy quality or guarantee safety outside the observed maps.
- One training seed, fixed evaluation seat and25 maps, entropy-seeded training opponents; no claim of robust superiority or bitwise reproducibility. No stage4 or extra seeds were started.

### Code and verification

Code HEAD11152fef7553a7650b95be99b5c8958149df6c44 plus the new hunting harness/config/tests. Windows11, Python3.12.10, NumPy2.5.2, Torch2.14.0, one Torch CPU thread; exact dependency/source hashes are in the results JSON/environment.json. The harness isolates evaluation opponent RNG streams and records actual feature extraction; agent reward/mask/encoder/algorithm and shared driver were unchanged.

Before the run:4 focused tests passed; full pytest **919 passed** (172.96s); Ruff check and format (146 files) passed; Pyright0 errors/warnings; git diff --check passed. Only the report and derived summary were added after completion, so training and the full suite were not repeated. Final whitespace check passed. No commit/push/merge.

