# D12 final held-out evaluation

## Execution record, fixed before held-out results

2026-09-21, branch `feature/ivan-dqn`. Preparation commit `fe8b23a` is pushed.
The user's authorization now includes the frozen held-out schedule, final
submission selection, archive, PR, review and post-merge checks. No training,
hyperparameter search or changes to inference policies are included.

The unchanged [protocol](d12_frozen_protocol.json), SHA256
`05e091bd4c8653585cf677feabf050bb621d10ceef5f6d830720e55b3f679ff0`,
specifies 17400 quality-evaluation games. Two practical candidates are eligible
for submission; six D8 finals are a separate training-seed diagnostic panel.
All models and inference modules are checked against frozen SHA256 values.
Execution uses the immutable Linux image from [runtime lock](d12_runtime_lock.json),
two CPU processes, one BLAS thread each, no network and read-only model mount.

Latency is a separate jobs1 pass: both practical candidates against three rules,
seat0, seeds3000–3024 (50 repeats). These games do not enter strength summaries
or model selection. Report pooled per-decision p99/max and setup separately.
This fixes the latency sample before looking at held-out scores.

The runner creates a fresh world per game and private opponent RNG streams by
actual seat. Stock entropy-seeding setup cannot alter caller RNG. Candidate RNG
is500; exact loaded arrays and absence of trainer/Torch are asserted. All paired
arms must have matching initial arena/coin/position hashes. Completed game files
are reused on resume; a failed game is retained and requires review, never silently
dropped or automatically replayed. Timeouts and 400-step limits remain results.

Smoke validation used only seed500. It exposed late-bound seat iteration in the
new harness's mixed-lineup wrapper; eager seat assignment fixed it, with a
regression test. No held-out games had started. Evidence from the failed attempt
is preserved under `smoke_attempt1`; the corrected smoke completed78 games with
matching paired boards and no harness failures. Neither agent code nor frozen
weights changed.

```powershell
.venv/Scripts/python.exe -m docs.experiments.d12_execute --out results/dqn/d12_heldout_20260921 --mode heldout
.venv/Scripts/python.exe -m docs.experiments.d12_execute --out results/dqn/d12_heldout_20260921 --mode latency
```

Raw files: `results/dqn/d12_heldout_20260921/`. Model copies are byte-identical
to frozen sources. Per-game records retain all agents and decision latencies.
No old validation score is reused as held-out evidence. D6/D7 failures and
reduced-D8 curriculum limitations remain unchanged.

## Results

Pending execution; no submission winner is declared in this preregistration.
