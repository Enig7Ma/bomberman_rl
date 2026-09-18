# Optional 3-step returns: implementation and preparation only

2026-09-18, branch `feature/ivan-dqn`. This is the explicitly agreed early D10
"Delayed bomb credit" experiment from the local plan
`C:/Users/ivans/Downloads/Telegram Desktop/dqn.md`. No full training run, stage3/4
or hyperparameter search was performed. D6/D7 results are unchanged; neither
stage-2 DQN final model passed its original threshold. Working lr remains3e-4.
Base code commit: `60cb65d472dd1d745b3ddf2bba0519469671f2eb`, plus the changes
listed in this task. The working tree was clean before implementation.

The hypothesis is that propagating observed consequences over several steps
helps learning. Delayed credit is not an established cause of the DQN deficit.
The only experimental setting change is `n_step=1` → `n_step=3`; rewards, mask,
encoder, lr, target_every and curriculum remain unchanged. The default is **1**.
The prepared JSON differs from `dqn_d7_stage2.json` only in its name and n_step.

## Data and targets

`nstep.py` is DQN-only. It consumes completed transitions from the unchanged
Bookkeeper, owns copies of their arrays and keeps at most two pending starts.
It checks consecutive IDs, round/stage, successor input and potential before
extending a sequence. A terminal flush emits every remaining start, including
the last action. No sequence crosses a terminal or round boundary.

For each start, replay retains its canonical x/action, the final successor's
independently canonical x_next/mask_next, length k, and up to three original
component tuples `(base, crates, deaths, phi, phi_next)`. It does not simply add
crate/death counts, and does not freeze a reward computed with today's aids.
The legacy scalar fields describe the first transition's base/counts/phi and
the final successor's phi; the `components` array is authoritative for the
multi-step return. At sampling time:

```
r_j = base_j + crate_aid*crates_j + death_aid*deaths_j
      + c_coin*(gamma*phi_next_j - phi_j)
R   = sum(gamma**j * r_j for j in range(k))
y   = R                                     if terminal
      R + gamma**k * Q_target(x_next, a*)    otherwise
a*  = masked argmax Q_online(x_next, action)
```

The terminal successor potential is zero. Internal shaping terms telescope
when successive potentials agree, but storing the original components also
allows a direct check against discounted `Rewards` calculations. Gamma and aid
coefficients are applied at sampling time. No intermediate action is replaced
with a hypothetical greedy action. Terminal rows never enter next-state argmax
or either next-state network. Padding contributes zero. The one-step reward
arithmetic remains the previous path; its emitted records match direct insertion.

## Counters, delayed readiness and persistence

Each real completed action increments transitions and stage_position **once**.
Updates remain scheduled by that real counter and warm-up/train_every, not by
the number of rows emitted. Flushing three rows does not cause three updates.
Epsilon still uses real stage transitions. In a new 3-step run the first two
nonterminal actions do not yet emit rows. A scheduled update with an empty replay
is skipped and recorded in `skipped_updates`; it is not caught up later. With
warmup=5000 this startup delay has passed before the first update. A D6 fork has
an already populated replay, so it skips no updates for readiness. Learner still
alone counts updates and synchronises target.

Replay and checkpoint writes use format **2**. Replay v1 is explicitly upgraded
to k=1 with its original component tuples; it is not relabelled as k=3. The
loader checks horizon, dtype/shape, components, terminal normalization, chronological
IDs, ring length/write position and generation. An ordinary horizon-changing
resume is rejected. V1 checkpoints without n_step mean n_step=1. Old readers
reject v2 rather than silently interpreting it as v1.

Checkpoint includes the pending queue as builtin lists/scalars, compatible with
`torch.load(weights_only=True)`, plus skipped update count. Replay metadata counts
emitted rows separately from real transitions when a queue exists. Tests round-trip
a nonempty queue through the actual Trainer checkpoint/replay pair. Standard
framework saves remain at round boundaries, where the queue is empty. This does
not add arbitrary mid-game engine/world restoration; the existing Bookkeeper
in-flight-action save guard remains. A stage change also rejects an unflushed queue.
Per-file atomicity and the existing explicit stale-replay limitations are unchanged.

## Comparable start: actual D6 audit

Source is the **same full parent used by the original stage2**, not D7 or a100k
snapshot:

`results/dqn/d6_20260917/pilot_lr3e-04_target1000/run_0/`

| Parent file | SHA-256 |
|---|---|
| checkpoint.pt | 4de4b2cfae5b0794e6ab4cc53b0c126c6d590afd057ef412013b9c47139811cd |
| replay.npz | 4fe49c723bd3989b00ca77b0e16aa7e2ec6be5d2b2eb2f424c1fb0a01a7c06ed |
| q_net.npz | ba1b3e407bff0de6e86d47bd95c7fd3ad6613d5ce83f5e678dc44f1b2c769e98 |

Strict conversion succeeded for **all50,047 retained starts**, IDs0–50,046,
319 complete episodes. The ring contains the whole parent history here; no
gaps or unfinished live tail were found. Conversion preserves the same physical
slots, start x/action, round/stage and transition IDs. A wrapped ring is also
supported: its oldest retained start needs only successors, not its overwritten
predecessor. Missing IDs, invalid boundaries/successors or an unfinished trailing
episode cause failure, not truncation, invented returns or a replay reset.

| Prepared sequence length | Records |
|---|---:|
| 3 | 49,409 |
| 2 (terminal tail) | 319 |
| 1 (terminal tail) | 319 |
| Total | 50,047 |

The preparation script called the existing `fork_curriculum` with explicit
`convert_n_step=True`, saving a separate child at
`results/dqn/d10_nstep3_prepare_20260918/`. The child retains **50,047 global
transitions,11,262 updates,319 rounds**, all online/target tensors, Adam state and
RNG states. It resets only the new stage's epsilon position, as the baseline
stage transfer already did. Actual Adam lr is **0.0003**. No games or updates
were run by preparation. Parent hashes before/after agree. The actual audit is
[dqn_d10_nstep3_preparation.json](dqn_d10_nstep3_preparation.json).

The one-step control keeps its original D6 replay; the experimental arm uses
the same starts and observed trajectories with correctly reconstructed returns.
Neither arm clears replay. The changed horizon necessarily changes which future
rewards/successors appear in its targets; that is the intended experimental
difference. Identical later trajectories are neither required nor expected.

## Future comparison and limitations

The prepared curriculum retains seed0,300,000 additional transitions with round
completion,40/40/20 loot/classic/coin mixture, lr3e-4,target_every1000, epsilon
0.3→0.05 over the first60%, probe_v2 and the original snapshot/validation protocol.
The reference final loot result remains28.4; improvement over it and exceeding
the unchanged stage2 threshold (>43.05 coins,0 suicides) are separate outcomes.
Navigation retention and paired per-map outcomes must still be reported.

Ordinary multi-step returns with replay are **off-policy biased**: intermediate
actions came from older exploratory behavior, not necessarily today's greedy
policy. No importance weighting, Retrace or other correction was added. D10
explicitly acknowledges this limitation. A single training seed would not prove
robust superiority. This implementation/preparation is not evidence that3-step
improves performance, and no full training comparison has been executed.

## Reproduction and checks

The following executed command prepares and verifies state only. It requires a
new empty destination and never invokes train_run or plays a game:

```powershell
.venv/Scripts/python.exe -m docs.experiments.dqn_nstep_prepare --parent results/dqn/d6_20260917/pilot_lr3e-04_target1000/run_0 --out results/dqn/d10_nstep3_prepare_20260918
```

It refuses to overwrite the already prepared directory. The child can later be
resumed by the existing training driver with `dqn_d10_nstep3.json`; **no training
command was executed in this task**. Tiny synthetic/engine test episodes are
unit/regression checks, kept separate from the prepared experiment.

```powershell
.venv/Scripts/python.exe -m pytest tests/test_dqn_agent_nstep.py -q
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check
.venv/Scripts/python.exe -m ruff format --check
.venv/Scripts/python.exe -m pyright --pythonpath .venv/Scripts/python.exe
git diff --check
```

Directed checks: **22 passed**. Existing related replay/learner/training/persistence/
continuation checks: **96 passed**. They cover manual masked Double-DQN targets,
terminal at positions1/2/3, discounted Rewards under four coefficient sets,
canonical endpoints, episode separation, copy ownership, one-step equivalence,
pending checkpoint, wrapped conversion/gap rejection, explicit legacy migration,
real counters with warmup0, full-state fork and subsequent resume.
Full suite: **915 passed in170.01s**. Ruff check and format (140 files), Pyright
(0 errors/warnings) and git diff --check pass. The latter only reports the
repository's LF→CRLF conversion notice, with no whitespace errors.

Bookkeeper, shared modules, tabular agent, reward/mask/encoder implementations and
game callbacks were not changed. Checkpoints/replay remain under ignored results/.
No commit, push or merge was performed.
