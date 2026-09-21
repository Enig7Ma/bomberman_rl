# Reduced D8: transition-budget preparation, 2026-09-20

Historical preparation note. The subsequently authorized six-run study and its
results are documented separately in [d8_reduced_protocol.md](d8_reduced_protocol.md)
and [d8_reduced_results.md](d8_reduced_results.md). Statements below about work
remaining/uncommitted describe the preparation checkpoint, not the final status.

The previous candidate audit was committed and pushed as `bcf35a6`.
This subsequent change prepares the driver; it does not run the proposed D8
training or change the negative D6/D7 conclusions. No stage4, dense encoder,
CNN, parameter search or held-out evaluation is included.

## Implemented

Tabular curricula can opt into `transitions` for every stage. Existing round
curricula retain their original driver and schedule. Mixed units within one
run are rejected. `rounds`, if supplied with transitions, is an optional hard
round cap for technical tests, as in DQN. The prepared study configs omit it.

`training/tabular_steps.py` samples tasks with the same per-chunk world-seed
rule as the DQN driver. The schedule uses stage-relative actual agent actions;
epsilon is set before each action, not once per chunk. A stage finishes its
current round when the transition budget is reached. The actual overshoot,
global transitions and final epsilon are in chunk records. Snapshots are made
on chunk boundaries crossing the stage-relative interval and at stage ends,
named with the actual global transition count.

Q values, visit counts, private Random state and the curriculum cursor/committed
chunk records share the existing atomic Q-table file. Automatic per-round model
saves are disabled in this managed mode. On restart, uncommitted metric tails
are trimmed and chunk/snapshot files reconstructed from that committed table.
A failure before model replacement repeats the interrupted chunk; a failure
after replacement resumes beyond it. This does not make entropy-seeded stock
opponents deterministic. Resume equivalence tests use controlled solo worlds.

`--init-from` is a new curriculum from copied weights/visits: it deliberately
removes the old driver cursor and initializes a new policy RNG. Reopening the
same run directory is resume, preserving its RNG/cursor. Parent models are
never modified. Frozen opponents use the existing snapshot mechanism.

Evaluation now recognizes transition snapshots for both agents and constructs
the table's transition axis from committed chunk counts and round steps. The
existing tabular metrics schema remains readable by `read_records`; no new
field was added to the agent's round record. Agent code, reward, mask, encoder,
Bookkeeper and learning updates are unchanged.

## Prepared configurations, not executed

- `d8_reduced_tabular_q_agent.json`
- `d8_reduced_dqn_agent.json`

Both specify50,000 navigation plus300,000 crates transitions, chunk5,
epsilon0.3 to0.05 over the first60% of each stage, and100k snapshot intervals
plus stage endpoints. Crates uses40% loot solo /40% classic solo /20% coin-heaven
per chunk. Both use E3/best_tier/canonical, gamma0.99, coin potential0.5 and
zero crate/death/bomb/spot aids, no teacher data. DQN uses onehot_e3, lr3e-4,
target_every1000,n_step1. Parameter names `coin_potential`/`c_coin` and discrete
E3/onehot_e3 differ by agent API, not by the intended reward/information.

Independent runs should start fresh; do not silently treat old seed0 artifacts
as a matched arm under this new protocol. The reduced study still needs its
final preregistration: exact evaluation position/opponent RNG protocol,
initial-model snapshots, fixed probe checks, paired-map intervals, visit-count
bins and cross-training-seed analysis. The generic driver configs alone do not
deliver those experiment-level safeguards or complete D8.

Next bounded step: finalize that harness and check it on tiny isolated budgets
before authorizing the six350k-transition runs. Training estimates and excluded
full-D8 items remain in `dqn_completion_plan.md`. No full run is launched here.

## Verification

Tests cover parsing/round compatibility, every action's epsilon, round-end
overshoot, real counters, transition snapshots, completed-run no-op, interruption
before/after atomic save, private RNG/Q/visit equivalence after resume, partial
metrics tails, stage changes and evaluation axes with unchanged weights.
Full suite: **925 passed in195.20s**. After the final metrics-compatibility
adjustment, all **74 affected driver/config/DQN tests passed in47.74s**.
Ruff check/format and git diff --check pass; Pyright reports0 errors/warnings.
The two prepared configs parse with350,000 planned transitions each. All
candidate/source model checksums still match the previously committed manifest.

These preparation changes remain in the working tree, separate from the pushed
candidate-audit commit. No study run or model replacement was performed.
