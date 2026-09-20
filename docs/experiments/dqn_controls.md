# D7: bounded comparison with existing control policies

2026-09-18. This is **not completed strict D8**. D6 was not reclassified as
successful; neither DQN stage-2 final model passes the loot threshold. No further
learning-rate search: the working setting remains `lr=3e-4`.

## Protocol and provenance

Exactly **60 new evaluation rounds**: three controls × two scenarios × ten maps.
World seeds 500–509, solo seat 0, ordinary loot-crate/classic generation and
400-step limit. Every control starts each round with its own `random.Random(500)`;
this matches the policy seed in the archived DQN evaluations. All use train=False,
without training callbacks or updates. Initial field, position and visible-coin
hashes agree across the three controls on each scenario/seed. No held-out seeds
3000–3099 were used. No stage 3/4 or training was started.

The new controls are:

- **Safe-random:** existing DQN `policy="random"`, uniform choice among actions
  allowed by `best_tier`. The explicit DQN model loads successfully, but the policy
  deliberately ignores its Q-values; this is not a missing-model fallback.
- **Feature heuristic:** existing `tabular_q_agent/heuristic.py`, unchanged.
  It follows visible coins, bombs when permitted and useful for crates/attack,
  then follows crate/opponent directions, finally choosing a permitted action.
  No new heuristic or hidden information was added.
- **Trained table (reference only):** tracked
  `agent_code/tabular_q_agent/model/q_table.npz`, introduced by Maria Druz's
  `507662ec402c059af2c05bcfd6aff67e5cfabf11` ("Train agent", 2026-09-16).
  Explicit strict loading verifies the loaded Q and visit arrays. A random tie
  in an unvisited table entry is part of this policy, not a loader fallback.

The new script asserts identical shared feature/mask/symmetry/core source files,
E3 categories and `best_tier` for all controls. DQN encodes those categories as
`onehot_e3`; the table uses their discrete index. The fixed heuristic reads the
same categories directly. Greedy learned inference has epsilon=0; random and
heuristic use only their private policy RNG. Model paths and arrays are asserted
before each new round. SHA-256 before/after checks confirm **174 source artifacts
unchanged**, including models, training checkpoint/replay and archived results.

Code HEAD: `f16076e` on `feature/ivan-dqn`, plus the new script. The exact script
checksum, full table metadata, artifact checksums and per-map metrics are in
[dqn_controls_results.json](dqn_controls_results.json). Raw actions, positions,
engine results, protocol and logs are in `results/dqn/d7_controls_20260918/`.

The table metadata records 18,400 rounds / **6,586,885 training steps**, E3,
best_tier, symmetry enabled, gamma=0.9, coin_potential=0.5, bomb_aid=0.1,
spot_potential=0.2, crate_aid=death_aid=0, alpha_omega=0.7, alpha_min=0.01.
The final stage is `nav-refresh`, seed=100007, epsilon=0.05, teacher_share=0.
Those last values describe the saved configuration, not a verified complete
curriculum or history of teacher use. Full curriculum/training-seed provenance
is unavailable locally. DQN instead used gamma=0.99, bomb/spot aids=0 and roughly
350k global transitions with the documented D6→D7 curriculum. Thus the table is
a useful deployed-policy reference, **not a matched-budget/reward learner arm**.
Its inference epsilon here is zero, not its saved training epsilon.

## Reused DQN evaluations (zero new rounds)

The existing protocol is the same: seeds500–509, solo seat0, maximum400 steps,
private policy seed500, explicit NumPy snapshot, train=False, no exploration.
The original evaluator checked path and every loaded weight array. The present
script reads its engine JSONL and decision traces, and verifies scenario, seat
and seed sequence. These evaluations were not repeated.

| Label | Snapshot under results/dqn/ | Global transitions | Stage-2 transitions |
|---|---|---:|---:|
| D6 parent | d7_stage2_seed0_20260918/checkpoints/transition_000050047 | 50,047 | 0 |
| DQN 3e-4 | d7_stage2_seed0_20260918/checkpoints/transition_000350346 | 350,346 | 300,299 |
| DQN 1e-4 | d7_stage2_lr1e4_seed0_20260918/checkpoints/transition_000350443 | 350,443 | 300,396 |

Each row reads `evaluation/{crates-solo,classic-solo}.jsonl` and the corresponding
`*_decisions.json` under that exact snapshot. Different actual budgets reflect
finishing the last round; both stage-2 planned budgets were300,000. Original
D6/D7 conclusions and checkpoints remain intact.

## Results

Coins/crates/bombs/steps below are means over **all ten rounds**, including those
that reach the limit. Suicides are events / ten rounds. WAIT is an action count
over all decisions. Backtracks count triples of consecutive pre-action positions
`A,B,A` with `A != B`, independently within each round; waiting in one place is
not a backtrack. This captures immediate returns, not every possible longer loop.
`NA` means the full ten-map position trace was not recorded, not zero.

### Loot-crate solo

| Policy | Coins | Crates | Bombs | Steps | Suicides | WAIT / decisions | Backtracks / triples |
|---|---:|---:|---:|---:|---:|---:|---:|
| Safe-random | 11.5 | 43.3 | 37.8 | 400 | 0/10 | 1098/4000 | 391/3980 |
| Feature heuristic | 43.3 | 106.5 | 33.8 | 400 | 0/10 | 174/4000 | 322/3980 |
| Table (reference) | 47.1 | 115.2 | 39.1 | 400 | 0/10 | 522/4000 | 257/3980 |
| D6 parent | 20.4 | 48.5 | 42.7 | 400 | 0/10 | 1176/4000 | NA |
| DQN 3e-4 | 28.4 | 68.6 | 46.5 | 400 | 0/10 | 923/4000 | NA |
| DQN 1e-4 | 22.9 | 56.1 | 44.0 | 400 | 0/10 | 725/4000 | NA |

### Classic solo

| Policy | Coins | Crates | Bombs | Steps | Suicides | WAIT / decisions | Backtracks / triples |
|---|---:|---:|---:|---:|---:|---:|---:|
| Safe-random | 2.1 | 43.6 | 39.1 | 400 | 0/10 | 1122/4000 | 390/3980 |
| Feature heuristic | 9.0 | 122.4 | 40.7 | 348.9 | 0/10 | 220/3489 | 219/3469 |
| Table (reference) | 8.9 | 119.0 | 52.2 | 400 | 0/10 | 802/4000 | 161/3980 |
| D6 parent | 1.2 | 16.3 | 51.0 | 400 | 0/10 | 1294/4000 | NA |
| DQN 3e-4 | 2.6 | 30.8 | 56.1 | 400 | 0/10 | 1125/4000 | NA |
| DQN 1e-4 | 4.0 | 66.3 | 35.8 | 400 | 0/10 | 1961/4000 | NA |

Deaths, invalid actions and timeouts are zero in each listed ten-round sample.
That is no safety guarantee outside these samples. The shorter heuristic classic
rounds are retained with their actual lengths; all other rows reach400 steps.
The archived three-map D7 backtrack diagnostics were not substituted for missing
ten-map metrics.

### Coins on each map

Columns are world seeds, in identical order for every policy.

| Loot-crate | 500 | 501 | 502 | 503 | 504 | 505 | 506 | 507 | 508 | 509 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Safe-random | 10 | 9 | 13 | 11 | 9 | 18 | 7 | 16 | 17 | 5 |
| Heuristic | 47 | 42 | 44 | 41 | 41 | 40 | 43 | 44 | 46 | 45 |
| Table | 49 | 45 | 49 | 44 | 46 | 47 | 47 | 47 | 48 | 49 |
| D6 parent | 31 | 18 | 40 | 22 | 0 | 16 | 28 | 38 | 1 | 10 |
| DQN 3e-4 | 36 | 12 | 36 | 34 | 13 | 23 | 42 | 40 | 21 | 27 |
| DQN 1e-4 | 32 | 8 | 35 | 40 | 13 | 37 | 12 | 19 | 1 | 32 |

| Classic | 500 | 501 | 502 | 503 | 504 | 505 | 506 | 507 | 508 | 509 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Safe-random | 2 | 2 | 5 | 2 | 3 | 2 | 2 | 1 | 1 | 1 |
| Heuristic | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 |
| Table | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 8 |
| D6 parent | 2 | 0 | 1 | 3 | 0 | 1 | 1 | 1 | 2 | 1 |
| DQN 3e-4 | 1 | 7 | 3 | 2 | 3 | 3 | 2 | 1 | 3 | 1 |
| DQN 1e-4 | 3 | 5 | 1 | 3 | 8 | 2 | 4 | 2 | 8 | 4 |

## Interpretation and one proposed next step

At lr=3e-4, trained DQN exceeds safe-random by16.9 loot coins on average and
wins on10/10 paired maps. In classic the gain is only0.5: three wins, five ties,
two losses. The smaller-lr model has means above random too, but remains worse
on the primary loot metric than the working model. These are observations for
one training seed and one fixed random-policy seed, not robust superiority.

DQN does **not** beat the fixed heuristic: both learned models lose on all ten
maps in both scenarios. The heuristic achieves43.3 loot coins with fewer bombs
and waits than DQN3e-4. This shows that the current features and allowed actions
can support materially better behavior on these maps. It does not prove that
they are fully sufficient, nor identify a learner bug. The table supports the
same feasibility observation, but cannot establish superiority of tabular
learning because reward, discount, training budget and curriculum differ.

The comparison cannot isolate optimization, delayed credit, exploration/data
coverage, representation aliasing, the mask's restrictions or finite budget.
Shared features/mask control those inputs for the heuristic comparison; they do
not turn a hand-written policy into a matched learning experiment. D8 additionally
requires matched training conditions and multiple independent seeds. Neither
DQN final model meets the unchanged stage-2 criterion **>43.05 loot coins and
zero suicides**. Passing that numerical bar with the heuristic/table does not
make DQN stage2 successful.

**One proposal, not executed:** compare 3-step returns against the existing
1-step learner, an explicitly early **D10 delayed-bomb-credit arm**, before
continuing curriculum. Hypothesis: propagating reward across several actions
improves learned bombing/collection behavior. The heuristic's better use of bombs
motivates testing credit assignment but does not establish it as the cause.
Use the same D6 seed0 full parent state, lr=3e-4, 300,000 additional transitions,
same mixture, epsilon, rewards, mask, encoder and evaluation/probe protocol;
change only return horizon1→3. Preserve terminal handling and acknowledge the
off-policy bias described in D10. Compare final loot coins with28.4 on paired
maps, report per-map differences, suicides and retained50/50 coin-heaven
collection; keep the >43.05 threshold separate from improvement. Stop on the
existing Q checks. No grid, budget extension or second seed is proposed here.
This proposed change would require its own implementation tests; no such change
or training was made in this task.

## Reproduction and validation

From the repository root, the command actually executed once was:

```powershell
.venv/Scripts/python.exe -m docs.experiments.dqn_controls --out results/dqn/d7_controls_20260918
```

The script requires a fresh output directory and refuses to overwrite an
existing one. It records the started-round count before each game and caps it
at60. Do not rerun collection for this task: its entire evaluation allowance
has been used. Existing saved summaries can be inspected without playing:

```powershell
Get-Content docs/experiments/dqn_controls_results.json -Raw | ConvertFrom-Json | Select-Object -ExpandProperty summaries
.venv/Scripts/python.exe -m pytest tests/test_dqn_controls.py -q
.venv/Scripts/python.exe -m ruff check
.venv/Scripts/python.exe -m ruff format --check
.venv/Scripts/python.exe -m pyright --pythonpath .venv/Scripts/python.exe
git diff --check
```

Two directed tests pass: backtracks distinguish waiting from immediate returns,
and aggregation retains failed maps and missing metrics rather than replacing
them with zeros. Ruff check/format (136 files), Pyright (0 errors/warnings) and
git diff --check pass. Full suite: **893 passed in173.63s**, including both new
tests. No commit, push or merge was performed.
