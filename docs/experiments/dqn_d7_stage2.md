# D7 — stage 2, seed 0

## Decision and protocol recorded before training

D6 remains unsuccessful under its original preregistered criteria. No pilot
winner was selected. The user authorized continuation after the D6 audit:
navigation improved, and all 36 archived networks passed the expanded probe's
numerical check. `lr=3e-4`, `target_every=1000` is a working baseline, not a
demonstrated winner. Loss need not decrease monotonically; it is diagnostic,
not a new mandatory acceptance filter.

Source: `C:/Users/ivans/Downloads/Telegram Desktop/dqn.md`, sections 5.4–5.9
and D7. The separately referenced tabular Q7 document was not found in the
repository (including ignored dev/) or Downloads; its curriculum and acceptance
quoted in D7 are used. No additional Q7 requirements are invented.

One training seed (0), 300,000 **additional** transitions, complete the last
round. Stage 2 has internal stage index 1. Global counters continue from D6's
50,047 transitions and 11,262 updates; stage epsilon position restarts at zero.
Epsilon is 0.3 → 0.05 over 180,000 stage transitions, then 0.05 (plan §5.4).
Keep onehot_e3, best_tier mask, gamma=0.99, c_coin=0.5, crate/death aids=0,
Adam, target interval, replay size, warmup and update cadence unchanged.

Interpretation of “50/50 + 20%”: **40% loot-crate solo, 40% classic solo,
20% coin-heaven solo**. These are probabilities of choosing a scenario for a
five-round chunk, not quotas of transitions. Realized chunk, round and transition
shares will be reported. A fresh curriculum cursor is created by the explicit
full-continuation API; no parent chunk history is edited. The driver measures
snapshot spacing from the start of the stage, on actual chunk boundaries.

Parent: `results/dqn/d6_20260917/pilot_lr3e-04_target1000/run_0/checkpoint.pt`.
SHA256: `4de4b2cfae5b0794e6ab4cc53b0c126c6d590afd057ef412013b9c47139811cd`.
Full replay, online, target, Adam, private RNGs and counters are restored through
the existing strict persistence loader. A new run ID identifies the branch;
the diagnostic probe is explicitly replaced only at this stage boundary.
Ordinary resume still rejects a config/probe mismatch or stale replay.

Fixed probe: `results/dqn/d6_audit_20260918/probe_v2.npz`, 2,000 rows,
417 unique encoded states. SHA256:
`008253a6aa52af1e006b0d3858591f4a0cc94c03729073e659fb7ce010e3189f`.
Measure initially, at snapshots, and every 10,000 global updates. Nonfinite Q
or any |Q|>50 stops this run, without an automatic retry or replacement seed.

Evaluate initially and around each 100,000 stage transitions on seeds 500–509,
crates-solo, classic-solo and coin-heaven-solo. Explicit archived NumPy weights,
train=False, no exploration/updates, no fallback; verify loaded arrays and file
checksums. The inference action RNG uses seed 500, as in D6. World seeds are the
ten evaluation seeds; training world seeds use the driver range 100000+chunk.

Stage criterion: mean loot-crate coins >43.05 and zero suicides in these 10
rounds. Report classic and navigation separately. Zero suicides/10 is not a
safety guarantee. Coin-heaven retention: compare all-50 completion and all-round
steps with D6's 10/10, mean124.7, range117–132; D6's original per-round126
failure remains in the record. Counts and per-agent-round rates use denominator
10 per snapshot/scenario; action frequencies use total agent actions (reported).
No stage 3/4, other training seeds or budget extensions are authorized here.

## Commands

```powershell
.venv/Scripts/python.exe -m docs.experiments.dqn_d7 --smoke --out results/dqn/d7_stage2_smoke_20260918
.venv/Scripts/python.exe -m docs.experiments.dqn_d7 --out results/dqn/d7_stage2_seed0_20260918
```

Each command needs a fresh directory and never overwrites a prior experiment.
Smoke uses a separate full branch and at most two rounds (600-transition target,
with round completion). It completed 800 transitions/200 updates and a subsequent
resume did no further work. The parent checksums remained unchanged. The main
run starts from D6 again, not from this smoke. The regression test compares all
learner state and RNG fields exactly before any training on the new branch.

Raw files and full archives are under results/dqn/. Config, source and summaries
are tracked; learned weights/replay are not. Throughput excludes evaluation but
includes training startup, world reconstruction, diagnostics and full saves.

## Results

**Stage 2 failed its coin criterion.** The final loot-crate mean is **28.4**,
below the strict requirement **>43.05**. Suicides are **0/10**. The navigation
skill is retained on the ten D6 validation maps. D6 remains unsuccessful under
its original rule; no pilot winner is retroactively selected.

One planned run completed, with **300,299 additional transitions**, **75,075
additional updates**, 859 rounds and 172 chunks. Global totals are **350,346
transitions**, **86,337 updates**, 1,178 rounds. The last round accounts for the
299-transition budget overrun. Epsilon ends at0.05; replay is full at100,000.

Measured training/diagnostic/save time: **915.311s (15m15.3s)**;
evaluation: **51.760s**; combined timed interval: **967.071s (16m7.1s)**.
Speed: **328.084 transitions/s**, **82.021 updates/s**. These are end-to-end
throughputs, not isolated learner timings. The timer includes world startup,
checkpoint restoration between chunks, full saves and probes, but excludes
the initial fork/preparation before the timer and excludes evaluation from the
training denominator. There was no new warm-up: global counters continued from
D6. One Torch CPU thread; no concurrent training run or hyperparameter search.

### Learning curves on fixed validation maps

Each row uses the same seeds500–509, ten rounds per scenario. Entries below are
means per agent-round; failed/incomplete rounds are included. Raw score equals
coins in these solo evaluations (no opponent kills).

| Stage transitions | Global transitions | Global updates | Loot coins | Loot crates | Loot bombs | Classic coins | Classic crates | Classic bombs | Coin-heaven coins / steps |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 50,047 | 11,262 | 20.4 | 48.5 | 42.7 | 1.2 | 16.3 | 51.0 | 50 /124.7 |
| 100,570 | 150,617 | 36,405 | 2.1 | 10.5 | 4.0 | 0.9 | 20.9 | 12.2 | 50 /124.7 |
| 200,607 | 250,654 | 61,414 | 26.5 | 66.7 | 47.5 | 3.2 | 47.1 | 55.8 | 50 /124.7 |
| 300,299 | 350,346 | 86,337 | **28.4** | 68.6 | 46.5 | **2.6** | 30.8 | 56.1 | **50 /124.7** |

Loot and classic always reached the400-step limit; no unfinished round is dropped.
Every snapshot/scenario has deaths0/10, suicides0/10 and timeouts0/10 agent-rounds.
Invalid actions are0/4,000 agent actions in each loot/classic evaluation and
0/1,247 in each navigation evaluation. At the final snapshot BOMB/action is
465/4,000 (11.625%) for loot,561/4,000 (14.025%) for classic,0/1,247 for navigation.
Navigation is50/50 coins on10/10 rounds at every snapshot. Steps are exactly
`131,120,125,121,130,117,120,125,126,132`: mean124.7, min117, max132,
7/10 rounds≤126. Retention does not turn the original D6 criterion into a pass.
Zero suicides in ten loot maps is an observation, not a safety guarantee.

### Actual mixture

The preregistered weights apply to chunks, not transitions. Shorter coin rounds
therefore contribute fewer transitions than their share of chunks.

| Scenario | Chunks | Chunk share | Rounds | Round share | Transitions | Transition share |
|---|---:|---:|---:|---:|---:|---:|
| loot-crate | 65 | 37.79% | 324 | 37.72% | 129,600 | 43.16% |
| classic | 69 | 40.12% | 345 | 40.16% | 138,000 | 45.95% |
| coin-heaven | 38 | 22.09% | 190 | 22.12% | 32,699 | 10.89% |

### Diagnostics

All scheduled probe checks passed; snapshot max|Q| was4.2107,11.4585,17.0689,
21.5701. Checks at global updates20k,30k,40k,50k,60k,70k,80k had maxima
6.5932,9.5194,12.4089,14.5430,16.9130,18.6280,20.1841. No NaN/Inf or Q-bound
stop occurred. The largest recorded fixed-probe value is **21.5701<50**.
Per-step training guards also stayed active. This does not certify every possible
game state. Policy churn is measured against the immediately preceding probe,
including periodic checks: at the final snapshot234/2,000 rows (11.7%) changed
greedy action. Duplicate rows retain the original probe weighting.

Comparable early/late windows use the first/last20 completed training rounds,
weighted by their actual update counts (1,741 and2,000). No zero-update row
enters the average:

| Quantity | First20 | Last20 |
|---|---:|---:|
| Huber loss | 0.092311 | 0.051875 |
| Mean absolute TD error | 0.336227 | 0.157162 |
| Gradient norm before clipping | 0.449150 | 0.458179 |

These observations are not a monotonic-loss filter. Lower loss coexisted with a
severe validation dip near100k. Loot performance recovered and improved over
the parent by8.0 coins/round, but did not reach acceptance. Classic performance
was also non-monotonic; no earlier snapshot is substituted for the planned final.

At the final snapshot the mean allowed-maxQ minus realized discounted shaped
return per decision is−2.3508 (loot),+1.8604 (classic),+5.2642 (navigation).
Mean absolute gaps are3.8316,1.8739,9.0369 respectively. These finite-trajectory
comparisons show imperfect value calibration; they alone do not establish
numerical divergence or its absence.

In loot,220 of465 BOMB actions (47.31%) occur with bomb_yield=0 and attack=none;
this is220/2,709 such decision contexts (8.12%). Classic:451/561 BOMB actions
(80.39%), or451/3,326 eligible contexts (13.56%). Final WAIT frequencies are
923/4,000 loot,1,125/4,000 classic,2/1,247 navigation actions. These are concrete
targets for a bounded follow-up analysis of policy and delayed credit, not proof
of an implementation defect and not permission to change reward mid-experiment.

Training itself recorded0 deaths,0 suicides in859 rounds and0 invalid actions
in300,299 actions. It collected19,008 coins, destroyed44,644 crates and placed
27,829 bombs. No death post-mortem was available because no deaths occurred.

### Artifact identity and conclusion

Raw root: `results/dqn/d7_stage2_seed0_20260918/`. Every learning-curve row is
linked to its own full archive in
[dqn_d7_stage2_results.json](dqn_d7_stage2_results.json), including checkpoint
checksums, evaluation totals/denominators, periodic probes and transfer checks.

| Global transitions | Archive directory under raw root | checkpoint.pt SHA256 |
|---:|---|---|
| 50,047 | checkpoints/transition_000050047 | c9223a159b3f57f4ac3f58ae4dec14db18eb6aa5d8e9bc060bfe54478d1afeb7 |
| 150,617 | checkpoints/transition_000150617 | 7419fbc2f9587c92387ba29c404f71e314e20e090be2760c50d56e6ce358c992 |
| 250,654 | checkpoints/transition_000250654 | 377e1dbf7ac66893b5d3dcad1e910b4bbda200fd066dc24e59c47e34860b287f |
| **350,346** | **checkpoints/transition_000350346** | **249ab5d1b4434a8879568d85158073e9329d1213e0c5846ec6eb4762ab9ae087** |

The last archive is the result of this stage, with checkpoint.pt, replay.npz,
q_net.npz, probe.json and evaluation data. The run-root checkpoint/replay is the
same final full save (all three files match the archive byte for byte). A final
strict reload confirms exact_history=True, consistent checkpoint/replay, replay
length100,000 and changed online weights. All three original D6 files retain
their recorded hashes.
The initial D7 archive was additionally compared directly with D6: every online,
target, optimizer and RNG field and every replay row/ring index matches. New
experiment identity, curriculum cursor, probe and stage epsilon position are
the deliberate differences.

**Do not start the other two training seeds on the stage-acceptance argument:**
the required >43.05 coins was not reached. Do not proceed to stage3/4 here.
Navigation and numeric checks passed, but neither substitutes for crate-task
performance. A bounded diagnosis of ineffective bomb choices and the100k policy
dip is justified before deciding on a separately authorized next experiment.
No new seeds, extra training, commit, push or merge were performed.

Initial evaluation: loot-crate mean20.4 coins, classic mean1.2 coins, no deaths,
suicides, invalid actions or timeouts. Coin-heaven reproduces D6 exactly:
50 coins in each of10 rounds, mean124.7 steps. Initial expanded-probe max|Q|=4.210668.

Evaluation also records each actual decision's allowed maxQ and its realized
discounted shaped return (gamma0.99, terminal potential0), including unsuccessful
rounds. Feature extraction is observed once, without resampling RNG. The signed
gap is a diagnostic comparison with this finite realized trajectory, not a proof
of over/underestimation of the optimal infinite-horizon value. BOMB is counted
separately where bomb_yield=0 and attack=none; raw decision logs preserve both
the numerator and number of such contexts.

Validation before the main run: 888 tests passed (168.91s), Ruff check and format
(130 files) passed, Pyright0 errors/warnings, git diff --check passed. Related
driver/continuation tests previously passed22/22; the full suite adds a directed
realized-return test and compares every replay row as well as ring metadata.

Code base:76953cf, with the source changes recorded by this task. Raw
environment.json contains hashes of the agent/training sources and the executed
experiment source is copied beside it. Windows11, Python3.12.10, NumPy2.5.2,
Torch2.14.0, pygame2.6.1, SciPy1.18.1; pytest9.1.1, Ruff0.16.7, Pyright1.1.414.

The executed D7 script matches the tracked working copy. Its SHA256 is
`8320ea3342de1721abf2f2d2138643d6182f1ce45dd53a4802ed901ebc701142`.

Changed source files: training/continuation.py (explicit full branch),
training/dqn.py (stage offset and stage-relative snapshot interval),
training/driver.py (public run metadata initializer), docs/experiments/dqn_d7.py
(bounded experiment and evaluation), tests/test_training_continuation.py
(full-state equality and realized-return regression). The configuration,
this report, compact JSON and the index link in docs/experiments/dqn.md complete
the tracked artifacts. No agent policy, encoder, reward or learner was changed.

## Bounded post-experiment diagnosis (2026-09-18)

This section is additional analysis, not a replacement of the original results.
No training, extra training seeds, stage3/4, reward/mask/learner changes or new
model selection occurred. The four original checkpoint/replay/model triples and
all original JSON/JSONL records were hashed before and after collection: unchanged.
Compact reproducible findings: [dqn_d7_diagnosis.json](dqn_d7_diagnosis.json).
Collector/auditor: [dqn_d7_diagnose.py](dqn_d7_diagnose.py).

### Is the100k dip real?

The table in “Learning curves” above retains all three scenarios separately,
actual stage/global counters, coins, crates and bombs. All loot/classic assessments
last400 steps and have0 deaths/10; all navigation assessments last124.7 steps on
average and have0 deaths/10. The parent D6 NumPy arrays match the initial D7 archive
exactly, so the zero-stage baseline is genuinely the original D6 policy.

On the ten paired loot maps500–509, coin changes from stage0 to100,570 are
`[-29,-18,-34,-22,+2,-13,-21,-38,0,-10]`: **8 worse,1 tied,1 better**, mean−18.3
coins (20.4→2.1,−89.7%). This is a broad deterioration across this validation
set, not one unlucky map. Classic changes are`[0,0,0,-3,+4,-1,0,-1,-2,0]`:
4 worse,5 tied,1 better, mean−0.3 (1.2→0.9); the evidence for a broad classic
regression is much weaker. Navigation coins and step counts are identical.

The24 instrumented replays reproduced the original coins, crates and bombs
exactly on maps500–502 for all four policies and both crate scenarios. Thus the
dip is reproducible for these fixed policies/maps/RNG settings. It is **not**
evidence of persistence across nearby checkpoints or independent training seeds:
neither was sampled. Indeed loot recovered to26.5 at200,607 and28.4 at300,299.
Final versus initial mean change is+8.0 loot coins and+1.4 classic coins, but the
final stage criterion still fails. No result is replaced by an earlier snapshot.

### Targeted configuration/resume checks

Recomputed from chunks: loot324/859 rounds (37.72%),129,600/300,299 transitions
(43.16%); classic345/859 (40.16%),138,000/300,299 (45.95%); navigation190/859
(22.12%),32,699/300,299 (10.89%). The distinction matters: the driver selects
scenarios by chunk, so20% requested chunk probability never meant20% transitions.

All859 round records satisfy `stage_position = transitions - 50047` and
`epsilon = .3 - .25*min(stage_position/180000,1)`. Epsilon is0.3 before the first
action,0.299444 after the first400-action round,0.0916667 at the stage midpoint
150,000, and0.05 from180,000 to the end. No per-chunk schedule reset or accidental
continuation of the already-finished D6 epsilon schedule was found.

Direct recursive comparison of the parent and initial archive confirms identical
online, target, Adam state, learner/feature RNG, global transitions and round count,
and identical replay rows/length/write position. New stage position0 and run ID
are deliberate. All subsequent global update counts satisfy
`updates = floor(global_transitions/4)-1249`; there was no second warm-up.

Strict persistence reload passes at all four archives. Replay sizes are50,047,
100,000,100,000,100,000; terminal-row counts319,294,280,285. All terminal next
potentials, next vectors and next masks are zero; all nonterminal next masks have
an allowed action. For every fully retained stage2 round (293,279,284 respectively
in the three trained archives), exactly one terminal is present, summed base
equals recorded engine coins/base reward, crate counts agree, and recomputed
shaped return agrees within1e-5. Partial rounds at the ring-buffer edge are not
mislabelled complete. Initial replay contains D6 rounds, not stage2 rounds.

At each archive,2,048 sampled transitions were checked under both the actual
coefficients and a second diagnostic coefficient set (no updates). Rewards are
recomputed with `base + crate_aid*crates + death_aid*deaths +
c_coin*(.99*phi_unit_next-phi_unit)` using the existing Rewards shaping function;
float32 batch rewards match exactly. The second set is only an arithmetic check,
not a changed experimental reward. Loader and replay tests also pass.

Every diagnostic world loads the explicit archived NumPy file; the collector
asserts its path and every weight array, train=False and absence of a trainer.
All9,600 captured actions are maximal-Q allowed actions after canonical mapping;
no fallback or exploration was used. Other snapshots are evaluated on the **same
captured vector and transformed mask**, without re-extracting random features.
No evidence of an implementation fault was found in these checked risks; this
bounded audit does not prove the whole implementation bug-free.

### Bomb behavior:24 paired diagnostic rounds

Four policies × two scenarios × three maps500–502 =24 rounds/9,600 actions.
One initial round completed before a NumPy-int JSON serialization error in the
new diagnostic collector; after fixing serialization it was replayed in the
24-round collection. **Total additional engine rounds25≤30**. The failed attempt
is retained separately; it changed no agent code or historical result.

Each table row pools3 rounds/1,200 decisions, all at the400-step limit with no
deaths. “Allowed” counts decisions whose mask includes BOMB. “Useful choice” is
BOMB selections divided by decisions with BOMB allowed AND bomb_yield>0.
“Destroying/placed” counts individual actual engine explosions destroying≥1
currently live crate, divided by placed bombs. Bombs still pending at the limit
are shown separately, not assumed to have failed or succeeded. Crates/bomb uses
all placed bombs. Empty contexts mean bomb_yield=0 and attack=none.

| Stage transitions | Scenario | BOMB allowed /1200 | Useful choice/opportunity | Destroying/placed | Pending | Crates/bomb | Empty BOMB/contexts | WAIT/1200 | Backtracks/1194 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | loot | 409 | 74/196 | 74/133 | 2 | 212/133=1.594 | 59/827 | 262 | 50 |
| 100,570 | loot | 1131 | 11/939 | 11/11 | 0 | 39/11=3.545 | 0/227 | 11 | **1111** |
| 200,607 | loot | 360 | 77/164 | 77/142 | 3 | 227/142=1.599 | 65/724 | 108 | 61 |
| 300,299 | loot | 355 | 69/156 | 69/141 | 1 | 197/141=1.397 | 72/838 | 296 | 13 |
| 0 | classic | 180 | 9/13 | 9/170 | 1 | 31/170=0.182 | 161/1008 | 283 | 131 |
| 100,570 | classic | 957 | 30/467 | 30/40 | 0 | 76/40=1.900 | 10/628 | 55 | **928** |
| 200,607 | classic | 183 | 45/53 | 45/169 | 0 | 105/169=0.621 | 124/776 | 73 | 81 |
| 300,299 | classic | 202 | 41/51 | 41/168 | 3 | 120/168=0.714 | 127/879 | 303 | 2 |

Immediate backtrack means decision positions A→B→A with A≠B; waits do not count.
Denominator3×398=1,194 position triples. This deliberately narrow measure misses
longer bomb/escape/wait cycles; few immediate backtracks do not imply efficiency.
Empty-bomb frequency among **placed bombs**, for loot snapshots, is59/133,0/11,
65/142,72/141; classic161/170,10/40,124/169,127/168. These25-round diagnostics
are separate from the original10-map evaluation frequencies above.

The100k agent has many allowed productive bombing opportunities but almost never
takes them in loot (11/939=1.17%). The mask is therefore not blocking discovery
at those decisions. Final loot takes69/156=44.23%, while many other bombs remain
unproductive. All9,600 recorded decision observations expose only the ordinary
agent-visible coins. Reachability is BFS on currently free cells with bomb tiles
blocked; it does not assert a temporally safe route. Immediately before the last
action, the three loot rounds have reachable visible coins totaling1,7,0,0 for
the four snapshots; classic totals0,3,0,0. A zero total does **not** mean all coins
were collected: some remain under crates or outside the current reachable region.

### Concrete decisions, identical observations across networks

Q below is mapped back to game action names; only allowed actions are shown.
Feature order is `(mask,coin_dir,crate_dir,bomb_yield,danger,opp_dir,attack)`;
directions0/1/2/3=UP/RIGHT/DOWN/LEFT,4=NONE,5=HERE. Raw observations, encoded
vectors, all Q values and engine outcomes are retained in the diagnostic JSON.

**A.100k, loot seed500, step25:** position(13,13), features
`(58,3,5,3,0,4,0)`, two reachable coins, nearest coin distance2. RIGHT is chosen
despite LEFT pointing toward a coin; BOMB is allowed and would hit live crates.

| Network (stage transitions) | RIGHT | LEFT | WAIT | BOMB | Greedy |
|---:|---:|---:|---:|---:|---|
| 0 | 2.671011 | 2.887399 | 2.572114 | 2.485607 | LEFT |
| 100,570 | **2.782829** | 2.765996 | 2.724675 | 2.430621 | RIGHT |
| 200,607 | 2.780698 | **2.832535** | 2.762505 | 2.479650 | LEFT |
| 300,299 | 3.086898 | **3.164440** | 3.090062 | 2.794489 | LEFT |

Step26 at(14,13) has features`(58,3,3,0,0,4,0)`, the same four allowed actions.
100k Q is RIGHT2.868398, LEFT2.877522, WAIT2.823788, BOMB2.683277, so it moves LEFT
back. All four networks prefer LEFT at this second observation. The100k network
then repeats these two positions through step400:376 actions, no additional
coin, no further explosion, final score2. Visible coins(11,13),(11,12) remain.
This directly demonstrates a learned ranking reversal and a greedy two-cell
cycle; it is not a claim that the mask or inverse symmetry transform failed.
The very small RIGHT–LEFT margins (0.01683/0.00912 at the two decisions) suggest
sensitivity, but do not by themselves identify the optimization cause.

**B.Final, loot seed500, step120:** position(13,9), features
`(63,4,3,0,0,4,0)`, no visible coin reachable, all six actions allowed.

| Network (stage transitions) | UP | RIGHT | DOWN | LEFT | WAIT | BOMB |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | −.151047 | −.178773 | −.100653 | −.198624 | −.132621 | **−.060226** |
| 100,570 | 1.267056 | 1.251580 | 1.174445 | 1.233127 | 1.252901 | **1.299030** |
| 200,607 | 1.639522 | 1.620905 | 1.654212 | 1.652396 | 1.633151 | **1.669242** |
| 300,299 | 2.079537 | 2.077491 | 2.071831 | 2.079980 | 2.076431 | **2.132963** |

BOMB is chosen by all four networks on this exact observation. The actual final
policy's bomb explodes at step124 and destroys **zero crates**. It waits/escapes;
score stays15 through step127, when it finally places a productive bomb elsewhere.
This useless bomb preference already existed before stage2; it is not solely
a newly introduced stage2 regression. No reward is paid for placing it.

**C.Final, classic seed500, step63:** position(15,9), features
`(61,4,2,0,0,4,0)`, allowed UP,DOWN,LEFT,WAIT,BOMB. Final Q respectively
`2.077574,2.078246,2.073012,2.071528,2.126488`: BOMB wins. On this same observation,
parent/100k/200k also prefer BOMB with Q0.008730/1.316494/1.673562 (full allowed
Q tables in compact JSON). Actual explosion step67 destroys0 crates; score stays1
through the next escape sequence. This supplies an independent scenario example
of the same inefficiency.

### What is established, and one proposed next experiment

Confirmed: a reproducible100k loot regression, greedy movement cycles, skipped
allowed productive bombs, unproductive actual explosions, and time lost to
wait/escape/repeated movement. Final navigation is retained. Training had37,728
exploration-marked actions/300,299 (12.56%) and27,829 bombs; **absence of exploration
is not supported**. Whether exploration was sufficient for delayed bomb credit
cannot be established without an intervention. Likewise onehot_e3 omits distances,
full geometry, remaining time and history, so aliasing is possible; these examples
do not prove that a richer encoder is necessary. Another network already chooses
the useful LEFT on exampleA with exactly the same features.

No specific agent implementation error was found, hence no speculative agent fix
was made. Numerical bounds pass, but policy quality and value calibration remain
imperfect. The observation supports investigating optimization-sensitive action
ranking; it does not prove learning rate is the root cause.

**Propose one controlled follow-up, not executed:** repeat this stage from the
same complete D6 seed0 state with **only lr3e-4→1e-4**. This is the alternative
already listed in plan§5.4/D6, but applying a new stage2 tuning trial before the
planned D9/D10 sequence is an explicit deviation in order, requiring a separate
authorization. Preserve inherited Adam moments/target/replay/RNG; change only
optimizer learning rate at the stage boundary. Keep seed0,300,000 additional
transitions with round completion, target_every1000, mixture/chunk boundaries,
epsilon, mask, reward, representation, fixed probe and validation maps unchanged.
The existing3e-4 run is the control, not a pilot winner; do not retrain it.

Hypothesis: smaller updates reduce the destructive ranking reversal/cycles near
100k without preventing useful crate-policy learning by300k. Predeclare comparison
at actual snapshot boundaries: primary final paired loot coins, with stage success
still strictly>43.05 mean and0/10 suicides; retain navigation50 coins on10/10 and
mean steps no worse than124.7 on these maps. Secondary mechanism check: at≈100k,
loot mean≥the parent's20.4 and fewer immediate backtracks than1111/1194 on
maps500–502 using the same harness. If only loss improves, reject the hypothesis
as unsupported by behavior. Report all snapshots and incomplete rounds; no extra
budget, second knob or automatic further seeds. A single matched training seed
would be a controlled diagnostic, not evidence of general superiority.

### Reproduction and checks

```powershell
# Collection requires a new output directory; do not rerun during this audit.
.venv/Scripts/python.exe -m docs.experiments.dqn_d7_diagnose --out results/dqn/d7_diagnosis_20260918_v2
# This command reads saved games/checkpoints only, and can be repeated safely.
.venv/Scripts/python.exe -m docs.experiments.dqn_d7_diagnose --analyze --out results/dqn/d7_diagnosis_20260918_v2
Copy-Item results/dqn/d7_diagnosis_20260918_v2/audit.json docs/experiments/dqn_d7_diagnosis.json
```

New collector tests distinguish allowed opportunities from prohibited BOMB,
placed bombs from exploded/destroying/pending bombs, and visible-coin reachability
from hidden coins. Related checks:56 tests passed; Ruff check/format and Pyright
pass. Full suite: **890 passed in87.17s**; final Pyright0 errors/warnings,
Ruff check and format132 files passed; git diff --check passed. Original checkpoints,
probe, experiment code and original results JSON are unchanged.
