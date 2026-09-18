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
