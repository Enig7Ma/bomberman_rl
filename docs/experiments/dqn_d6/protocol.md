# D6 preregistration (before training)

Code baseline: 1739492d4135823114b9e83abd7088e3540d88d4. Modified experiment
source hashes and dependency versions are captured for every run.

- Coin-heaven solo, onehot_e3, best_tier mask, unchanged rewards/defaults.
- Four combinations, in tie-breaking order: (3e-4,1000), (3e-4,250),
  (1e-4,1000), (1e-4,250). Pilot seed 0; 50,000 transitions each.
- Epsilon .3 -> .05 over first 30,000 transitions. Warm-up 5,000, batch64,
  train_every4. Five-round chunks; snapshots at crossed 10,000 thresholds
  and final stage boundary, using ACTUAL counts. Overshoot is retained.
- One fixed probe: 1,000 sampled observed states per source agent, bfs_agent
  and rule_based_agent, coin-heaven solo, seeds900..909. Sampling without
  replacement among observations; duplicates in encoded states retained.
  All source RNGs explicitly seeded; extraction and sampling RNG independent.
  Probe is read-only diagnostics, never inserted into training replay.
- Training seeds0,1,2 use independent initialisations. Validation500..509;
  probe900..909; held-out3000+ untouched. Two processes, one Torch thread each.
- Evaluate the untrained initial network and EVERY snapshot with identical
  validation seeds, train=False, learned policy, epsilon effectively zero,
  explicit NumPy model path. No safe-random fallback, no control arm.
- Stable eligibility: completed budget; no NaN/Inf or |Q|>50 abort; mean loss
  over last20 updated rounds <= 1.25 times first20 + 1e-6. This last test is
  a preregistered operational diagnostic for "loss not growing", not a plan
  threshold or a tuning score. Report its numeric values and failures.
- Select among eligible runs by highest final-snapshot mean coins, then
  highest complete-50 fraction, then lowest mean steps across ALL rounds.
  Exact ties use the combination order above. No best-intermediate-snapshot
  selection, and no loss-based ranking. If no run is eligible, report that
  selection failed rather than quietly choosing an excluded run.
- Reuse selected seed0; train only seeds1 and2 with identical selected config.
  Pilot is a limited settings check, not evidence of statistical superiority.
- Interpret <=126 steps strictly per validation round; report also all-round
  and successful-round means separately. All50 criterion requires all ten
  rounds on each of the three final seed models. Initial/intermediate results
  stay visible in learning curves. D-S7 checks probe finite/bounded plus loss
  eligibility on all three seeds. Initial and snapshot probes supplement
  existing every10,000-update probes. No threshold/reward/feature changes.
- On any numerical failure stop that run, keep failure text, last consistent
  checkpoint and failure weights if serialisable. Keep the seed in tables.
- End-to-end run timing includes training-world setup, full chunk saves,
  snapshot archives and probe checks; excludes validation and top-level Python
  startup. jobs2 throughput is per process under concurrent load.

No D7, full curriculum, commit, push or merge.


## Post-pilot diagnostic replication decision (not a successful selection)

All four seed0 pilots failed the preregistered growth filter; selection remains
null. The requested three-seed check is performed separately on the original
baseline lr3e-4/target1000, reusing seed0 and adding only fresh seeds1/2.
This decision was made AFTER seeing the pilot and must not be presented as a
preregistered successful configuration choice. It does not alter eligibility,
reward, feature, step cutoff, budget, or the four-combination search space.
Report these runs as diagnostic replication of a rejected baseline, retaining
all failures. No claim of stable selected settings or completed D6 acceptance.
