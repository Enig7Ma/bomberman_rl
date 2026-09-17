# DQN experiments

## D2 — NumPy inference, 2026-09-16

Plan: local `C:\Users\ivans\Downloads\Telegram Desktop\dqn.md`, sections
5.1, 5.6 and D2 (`dev/dqn.md` is absent). Branch: `feature/ivan-dqn`.
The worktree was clean at the start, after D1 at `d084f57` (660 tests).
No AGENTS.md or CLAUDE.md was found in the repository/ancestor directories.

Implemented `32 → 128 → 128 → 6` in NumPy float32, with ReLU after the first
two layers. Each output estimates one action's value. The agent extracts E3,
canonicalises it, encodes 32 inputs, takes the highest allowed Q-value and maps
the chosen action back. Ties use the private agent RNG; initial weights use a
separate NumPy RNG. `QFunction` separates callbacks from the future learner.

`q_net.npz` contains `W0,b0,W1,b1,W2,b2` and a scalar JSON `meta` with `header`
and `meta` objects. Loading validates schema, encoder, version, layer sizes,
activation, action order, float32 shapes, finite weights and metadata. Saving
uses a temporary file in the same directory, flush/fsync and atomic replace.
Outside training an explicit missing/broken path raises; the default path logs
an error and falls back to safe-random. The default policy is now `learned`;
`policy="random"` remains an explicit control. No trained model is shipped.

### Reproduce

From the repository root in PowerShell, using a **new** output directory:

```powershell
.venv/Scripts/python.exe -m docs.experiments.dqn_d2 --output results/dqn/d2_random_20260916
```

The checked-in runner creates one random network (`init_seed=0`, action
`seed=0`, encoder `onehot_e3`, mask `best_tier`, policy `learned`). It runs
`vs-rule-based`/`classic` against three `rule_based_agent` instances:

- Acceptance: seeds 500–524, four seat rotations, 100 rounds.
- Latency: seeds 300–309, four seat rotations, 40 separate rounds.
- Both use jobs=1 (one sequential world at a time), without training.
- Each round asserts that the explicit model is loaded, QNetwork is active,
  `policy="learned"`, and no Torch is imported. The file hash is unchanged.
- Latency includes the whole `act` callback, measured by the existing framework
  and `tournament.latency`; setup/loading is excluded. Mean is over every
  decision, p99 uses nearest rank, max is the largest observed decision.
- Opponents seed themselves from entropy, so results are not bit-reproducible.

Raw files (ignored by Git) are under `results/dqn/d2_random_20260916/`:
`acceptance.jsonl`, `latency.jsonl`, `summary.json`, `random_init_q_net.npz`.
The JSONL files retain per-round results, loaded-network assertions and every
DQN decision latency. The random weights are an experiment artifact, **not**
a final trained model. Metadata counters for training are all zero.

### Results

Environment: Windows 11 build 26200, Python 3.12.10, NumPy 2.5.2,
AMD64 Family 25 Model 68 Stepping 1. No other test/check processes ran during
the latency series. Framework timing uses its existing `time()` clock.

| Measurement (DQN only unless stated) | Acceptance | Separate latency run |
|---|---:|---:|
| Completed rounds | 100/100 | 40/40 |
| Loaded-network rounds / fallback rounds | 100 / 0 | 40 / 0 |
| Runtime errors | 0 | 0 |
| Timeouts (all four agents) | 0 | 0 |
| Suicides / rate | 0 / 0.000 | 1 / 0.025 |
| Invalid-action events | 351 | 116 |
| DQN decisions | 38,068 | 14,234 |
| Mean latency, ms | 0.645938 | 0.651927 |
| p99 latency, ms | 1.542568 | 1.512051 |
| Maximum latency, ms | 18.820524 | 18.025637 |

All D2 acceptance criteria **pass**: 100 rounds without errors, suicide rate
`0.00 < 0.10`, separate latency mean `0.652 < 2 ms`, max `18.026 < 50 ms`.
There are no unmet D2 criteria. Invalid-action events are recorded separately
from Python exceptions and timeouts; zero invalid actions is not a D2 gate.
The engine applies simultaneous decisions sequentially, so moves allowed in
the observation can become blocked before execution; this run does not retain
per-event traces to attribute each invalid action's cause.

This is a smoke/performance check of **untrained** weights, not a training
result or a claim of playing strength. No seed search or weight tuning was
performed. The experiment file is 81,089 bytes, with schema
`3963eb07bf146d11` and SHA-256
`a28e2754b9550b08f23cf3b91d0a58fd02da07980a1a5eb885dbe7ea43053631`.

### Code validation

Tests cover manual forward arithmetic, masked argmax, seeded ties, exact
weights/metadata round-trip, all six arrays' shapes/types/nonfinite values,
incompatible headers, loading rules, failed write/replace, canonical movement
and WAIT/BOMB, explicit random control and isolated inference without Torch.
Shared tabular files and their behavior are unchanged. No exclusions were
added to Ruff/Pyright. D3, replay buffers and training were not started.

| Check | Result |
|---|---|
| DQN network/callbacks/config/encoder/vendored tests | 289 passed |
| Full `pytest -q` | 736 passed |
| `ruff check` | Passed |
| `ruff format --check` | Passed |
| `pyright --pythonpath .venv/Scripts/python.exe` | 0 errors, 0 warnings |
| `git diff --check` | Passed |

Commands use `.venv/Scripts/python.exe -m` before pytest, Ruff and Pyright.
Pyright requires the explicit project interpreter in this shell; running it
without `--pythonpath` picked a different Python and could not resolve NumPy.
No dependencies or exclusions were changed to fix that environment issue.

### INVALID_ACTION diagnostic before the D2 commit

The original JSONL files contain aggregate invalid counts, not selected actions
or step snapshots. They establish 351/38,068 decisions (0.9220%, 88/100 rounds
affected) and 116/14,234 (0.8150%, 37/40 rounds affected), but cannot establish
which action or cause accounts for each of those 467 events.

Engine conditions (`environment.py`, `tile_is_free`, `perform_agent_action`,
`poll_and_run_agents`): a move requires arena value 0 and no bomb or active
agent on its destination **at execution time**. BOMB requires `bombs_left`;
WAIT is always accepted. A blocked move, unavailable BOMB or unknown command
produces INVALID_ACTION. Agents first decide from their observations, then the
engine applies their actions in a random permutation. A timeout instead
substitutes WAIT; it does not itself produce INVALID_ACTION.

Added opt-in instrumentation to `dqn_d2.py`, implemented in
`dqn_d2_trace.py`. It wraps the actual extraction/selection calls without
making extra RNG draws, verifies mask/action mappings on every decision, and
records invalid steps with the full original observation, features, original
and canonical masks, actual selected Q/action, symmetry, game action,
execution snapshot and earlier players' before/after moves. It does not change
callbacks, weights, masks or strategy. Diagnostic timings are not D2 latency
measurements.

```powershell
.venv/Scripts/python.exe -m docs.experiments.dqn_d2 --output results/dqn/d2_invalid_20260916_v2 --diagnose-model results/dqn/d2_random_20260916/random_init_q_net.npz --diagnose-seed 506
```

Only four complete rounds were replayed (seed 506, all seats, three rule-based
opponents), using the original file with unchanged SHA-256. A preliminary
attempt stopped at trace serialization of a NumPy scalar; the diagnostic
serializer was corrected before this complete run. The original 100+40 runs
were not repeated. Opponent entropy means this is a new sample on the same
seed, not an exact replay of the original decisions.

Results: 16/1,600 decisions (1.000%), affecting 3/4 rounds. Invalid actions:
UP 3, RIGHT 5, DOWN 4, LEFT 4, WAIT 0, BOMB 0. All 16 traces show a target free
in the observation, subsequently occupied by an opponent whose earlier move
into it is recorded. No cases remain unclassified in this diagnostic sample.
All 1,600 original/canonical mask and inverse-action assertions passed.

Concrete example: seed **506**, seat **1**, step **143**:

| Stage | Recorded evidence |
|---|---|
| Original observation | DQN at `(13,15)`; target `(12,15)` is floor, with no player or bomb. Opponent `rule_based_agent_2` is at `(11,15)`; its bomb is at `(11,15)`, timer 3. |
| Original mask | `RIGHT, LEFT, WAIT, BOMB` (bits 58) |
| Canonical transform | Three clockwise quarter-turns, no reflection |
| Canonical mask | `UP, DOWN, WAIT, BOMB` (bits 53) |
| Selected canonical action | `DOWN`, Q=0.041089; other allowed Q: UP=-0.023674, WAIT=-0.081594, BOMB=0.009967 |
| Inverse transform | `DOWN → LEFT`, destination `(12,15)` |
| Earlier engine action | `rule_based_agent_2`: `RIGHT`, `(11,15) → (12,15)` |
| DQN execution | Actual action `LEFT`; target still floor, now occupied by that opponent; `tile_is_free` is false, producing INVALID_ACTION. |

Thus action-order occupancy conflicts are **confirmed for this sample**;
there is no observed mask or coordinate-transform defect. The evidence does
not prove that every event in the older aggregate runs has this same cause.
No agent fix or strategy change is warranted by these findings. Full evidence
and counts are in ignored `results/dqn/d2_invalid_20260916_v2/`:
`invalid_traces.jsonl` and `summary.json`. The trace classifier can reanalyse
those files without running any games.

`test_dqn_agent_vendored.py` changed in D2 only to extend isolated subprocess
execution to both absent and loaded weights, proving the latter really loads
and still does not import Torch. `SHARED` and its byte-for-byte assertion are
unchanged: features, mask, symmetry, rewards, bookkeeping and all five core
modules (10 total) remain covered.

After diagnostic changes: callbacks/vendored tests **36 passed**, Ruff check
and format passed (103 files), Pyright 0 errors/warnings, `git diff --check`
passed. The full **736 passed** result above remains the D2 pre-diagnostic
run; production agent code was not changed during this diagnostic.

## D3, part 1 — replay buffer only

Implemented `agent_code/dqn_agent/replay.py` from local plan sections 5.4–5.5
and D3. This is **not completion of D3**. No learner, Torch training, callback
integration, checkpoint persistence or gameplay changes are included.

API: `ReplayBuffer(capacity, input_dim)`, `push(ReplayTransition(...))`,
`len(buffer)` and `sample(batch_size, rng, gamma=..., c_coin=...,
crate_aid=..., death_aid=...) -> ReplayBatch`. Capacity and input dimension are
positive integers; the caller supplies the encoder dimension (32 for OneHotE3).
All state vectors/actions/masks must already be in canonical coordinates;
replay stores them unchanged and cannot infer coordinate systems from vectors.

Decisions:

- Fixed NumPy structured storage, with all 13 fields and exact dtypes from
  section 5.5: float32 states/base/unit potentials, uint8 action/crates/deaths/
  stage, bool successor mask and terminal flag, uint32 round and uint64
  transition IDs. Death count is 0 or 1, including suicide's two death events.
- FIFO ring replacement: capacity 3 holding A,B,C receives D and retains
  B,C,D; the next insertion replaces B. Length stops growing at capacity.
- Uniform sampling **with replacement**, only from filled slots, using the
  supplied `numpy.random.Generator`. Duplicate samples are intentional; a
  batch can exceed the current length. Empty replay and nonpositive batch
  sizes are errors. Global RNG state is untouched.
- Inputs are validated before storage changes, then copied. Each batch owns
  independent contiguous arrays; mutating inputs or batches cannot modify the
  buffer. Wrong shapes/dtypes, nonfinite or overflowing values, invalid action
  indices/IDs, and empty nonterminal masks are rejected. Integer scalar inputs
  are Python ints (not bools); array dtypes must already match the schema.
- Terminal entries normalise `x_next`, `mask_next` and `phi_unit_next` to zero,
  even when valid nonzero placeholders were supplied. Nonterminal successor
  masks must allow at least one action. Unit potentials are in [0,1].
- Replay stores reward ingredients, never a cached total. Sampling computes
  `base + crate_aid*crates + death_aid*deaths + c_coin*(gamma*phi_unit_next-phi_unit)`.
  Float64 intermediates reduce cancellation; batch rewards are finite float32.
  Coefficients may change between samples without clearing replay. Comparison
  with shared `Rewards` uses its `coin_potential=c_coin`, `bomb_aid=0` and
  `spot_potential=0`; the latter two Q7 extensions are outside this D3 schema.

Validation: **52 replay tests passed**, including multiple wraps, capacity 1,
partial-buffer sampling, seeded reproducibility, approximate uniformity,
ownership, contiguous batches, dtype/ID boundaries, terminal handling, and
failed-input preservation of rows/write order. Reward tests use 100 generated
transitions, gamma values 0/0.99/1 and four coefficient settings on the same
buffer, compared with `Rewards` (including suicide and living round-end cases).

The existing `replay.py` exclusion matched the new nested file in Ruff. Both
check configurations now spell the legacy root path as `./replay.py`; no new
exclusion was added. `ruff check --show-files agent_code/dqn_agent/replay.py`
confirms the new module is checked. Pyright also reports/checks the new module.

Final validation: full pytest **788 passed**; Ruff check and format check
passed; Pyright with `.venv/Scripts/python.exe` reported 0 errors/warnings;
`git diff --check` passed. Existing gameplay and shared-module tests remain
unchanged. The rest of D3 is deferred to subsequent tasks.

## D3, part 2 — Double DQN learner, 2026-09-17

Implemented the learner and toy-data validation from the local plan's sections
5.4–5.6 and D3, on `feature/ivan-dqn` after clean base `41c7f0f` (788 tests).
No project AGENTS.md/CLAUDE.md was found; the plan remains at
`C:\Users\ivans\Downloads\Telegram Desktop\dqn.md`.

`learner.py` provides a CPU float32 QNet compatible with D2's
`32 → 128 → 128 → 6` network, ReLU hidden layers, a frozen initially identical
target, Adam (lr=3e-4, eps=1e-8), mean Huber loss (delta=1) and gradient norm
clipping at 10. The architecture stays fixed at 128–128 for D2 export
compatibility; alternative widths/heads are outside this task.

For a live successor the online network selects an action **within its mask**,
and the target network supplies that action's value. Terminal rows are handled
separately: neither next-state forward nor argmax runs for them and their
target equals reward exactly. Empty nonterminal masks are rejected. Loss uses
only the batch's selected action columns; target computation has no gradients.
Metrics return loss, mean absolute TD error and the gradient norm before clipping.

Counter ownership and randomness:

- The future caller owns total and stage-relative **transitions**, stage budget,
  replay insertion, warm-up (5000) and update cadence (one per 4 transitions).
  Calling `sample`, `select` or `epsilon` never increments any counter.
- Learner alone owns **updates**. A successful optimizer step increments it
  exactly once and automatically copies online to target every 1000 updates.
  The future caller must not repeat the sync from the plan's pseudocode.
- Epsilon is a pure function of caller-supplied stage-relative transitions:
  0.3 at zero, 0.175 after 30% of the stage, 0.05 at 60% and thereafter.
  `select(evaluate=True)` uses epsilon zero. Exploration and greedy ties use
  the learner's private Python RNG. Target argmax ties use the first allowed
  index deterministically; they do not consume the action RNG.
- Initialisation uses uniform +/-1/sqrt(fan_in), the Linear default
  distribution, with a private Torch CPU Generator (`init_seed`). Parameters
  are allocated directly so no nn.Linear constructor draws from global RNG.
  Replay sampling uses a separate private NumPy Generator (`seed`); negative
  Python seeds are supported via a private Random-derived nonnegative seed.
- Learner construction sets Torch CPU threads to 1. Tests verify that global
  Python, NumPy and Torch RNG states survive initialisation/selection and the
  500-update learning run unchanged.

`export_numpy` copies W0/b0/W1/b1/W2/b2; `Learner.export()` returns a D2
QNetwork snapshot with config/update metadata. Toy snapshots are labelled
`D3-toy`; future game-training callers must supply stage/transition/round
metadata. No learned weights are shipped. QNet implements `values(x)` for
future QFunction use, but callbacks remain untouched and NumPy-only.

Validation includes manual online/target disagreement and a larger forbidden
Q-value, terminal empty masks (even unused NaN successors), selected output-row
gradients, numerical Huber loss, clipping, target independence/sync timing,
epsilon boundaries, invalid batches and configuration validation.

- Five-cell corridor: gamma=0.9, one-hot inputs, LEFT masked at the wall;
  after 2500 updates (target interval 50 for this small test), greedy goes RIGHT
  in every cell and Q(RIGHT) matches `0.9^(4-cell)` within **0.05**.
- Overfit: 64 fixed distinct terminal transitions, 1000 updates with default
  learning rate; final mean Huber loss **<1e-3**.
- Export/save/load parity: 100 random numeric inputs plus 100 valid E3 inputs,
  Torch vs NumPy at **1e-5** tolerance; exported arrays are independent copies.
- Two learners with identical seeds and replay inputs produce identical
  samples, update metrics, online and target weights after **500 updates**.
- Torch tests use `pytest.importorskip("torch")`; installed **2.14.0+cpu**
  actually executed all **22** learner cases. No Torch cases were skipped.
- The existing isolated callbacks/setup/act tests still pass with and without
  loaded NumPy weights, and Torch remains absent from that process's modules.

### Update benchmark

Tracked script, invoked from the repository root with a fresh output filename:

```powershell
.venv/Scripts/python.exe -m docs.experiments.dqn_bench_update --output results/dqn/d3_learner_20260917/bench_update.json
```

Windows 11 build 26200, Python 3.12.10, NumPy 2.5.2, Torch 2.14.0+cpu,
AMD64 Family 25 Model 68 Stepping 1. Batch 64, 32→128→128→6, one CPU thread,
init/action seeds 0, fixed mixed terminal/nonterminal batch. Separate **200
warm-up** updates, then **1000 timed** updates via `perf_counter_ns`; no tests
or check processes ran concurrently with the benchmark. One automatic target
sync is included. Timings include validation, NumPy/Torch conversion, target,
forward/backward, clipping, Adam and metrics; they exclude replay sampling.

| Statistic | Measured update time |
|---|---:|
| Mean | 2.307977 ms |
| Median | 2.164200 ms |
| p95 | 3.010775 ms |
| Maximum | 4.231700 ms |

The **~0.5 ms reference was not reproduced** on this machine: mean is about
4.62 times that reference. No parameters or workload were tuned to approach it.
This is the remaining performance discrepancy; all functional criteria pass.
Raw samples/config/environment are in the ignored benchmark JSON above.

Final checks: **191 related tests passed**, full pytest **829 passed**, Ruff
check/format passed, Pyright (explicit venv interpreter) 0 errors/warnings,
and `git diff --check` passed. No new Ruff/Pyright exclusions or dependencies.
D4 game training callbacks, full-state checkpoints and curriculum are not
implemented. The only learning performed here was on toy/test/benchmark data.
