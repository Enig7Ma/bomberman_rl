# Reduced D8 results

Training/evaluation protocol: [d8_reduced_protocol.md](d8_reduced_protocol.md).
D6/D7 are not reclassified. This excludes hunting/combat training and all-seat head-to-head; full D8/D12 are not complete.

## Learning curves

| Agent | Seed | Actual transitions | Updates | Train seconds | Combat score [map95% CI] | Loot coins | Navigation coins / all50 |
|---|---:|---:|---:|---:|---|---:|---|
| tabular_q_agent | 0 | 0 | 0 | 0.3 | 1.08 [0.56,1.68] | 11.50 | 23.30 / 0/10 |
| tabular_q_agent | 0 | 50052 | 50052 | 54.4 | 1.32 [0.72,2.04] | 11.40 | 50.00 / 10/10 |
| tabular_q_agent | 0 | 151905 | 151905 | 165.1 | 1.88 [1.04,2.84] | 4.40 | 3.80 / 0/10 |
| tabular_q_agent | 0 | 250276 | 250276 | 268.7 | 3.08 [1.80,4.56] | 5.80 | 0.70 / 0/10 |
| tabular_q_agent | 0 | 350146 | 350146 | 374.5 | 2.16 [1.36,3.08] | 19.80 | 50.00 / 10/10 |
| dqn_agent | 0 | 0 | 0 | 4.1 | 0.00 [0.00,0.00] | 0.00 | 0.20 / 0/10 |
| dqn_agent | 0 | 50047 | 11262 | 122.8 | 3.64 [2.64,4.68] | 20.40 | 50.00 / 10/10 |
| dqn_agent | 0 | 150952 | 36489 | 391.6 | 1.92 [1.36,2.52] | 1.10 | 50.00 / 10/10 |
| dqn_agent | 0 | 250241 | 61311 | 663.1 | 2.64 [1.88,3.52] | 12.40 | 50.00 / 10/10 |
| dqn_agent | 0 | 350187 | 86297 | 936.8 | 2.84 [2.28,3.48] | 5.70 | 3.90 / 0/10 |
| tabular_q_agent | 1 | 0 | 0 | 0.3 | 1.08 [0.56,1.68] | 11.50 | 23.30 / 0/10 |
| tabular_q_agent | 1 | 50070 | 50070 | 49.4 | 1.32 [0.72,2.04] | 11.40 | 50.00 / 10/10 |
| tabular_q_agent | 1 | 150172 | 150172 | 152.7 | 1.80 [1.00,2.68] | 4.60 | 0.10 / 0/10 |
| tabular_q_agent | 1 | 250448 | 250448 | 255.5 | 2.36 [1.40,3.48] | 12.60 | 50.00 / 10/10 |
| tabular_q_agent | 1 | 350150 | 350150 | 360.7 | 2.16 [1.32,3.12] | 12.00 | 50.00 / 10/10 |
| dqn_agent | 1 | 0 | 0 | 4.1 | 0.00 [0.00,0.00] | 0.00 | 0.20 / 0/10 |
| dqn_agent | 1 | 50115 | 11279 | 117.7 | 2.32 [1.68,3.04] | 4.90 | 50.00 / 10/10 |
| dqn_agent | 1 | 150210 | 36303 | 359.2 | 3.20 [2.08,4.36] | 1.00 | 50.00 / 10/10 |
| dqn_agent | 1 | 250742 | 61436 | 613.2 | 2.84 [2.00,3.76] | 17.00 | 50.00 / 10/10 |
| dqn_agent | 1 | 350272 | 86319 | 889.7 | 2.84 [2.00,3.76] | 3.70 | 50.00 / 10/10 |
| tabular_q_agent | 2 | 0 | 0 | 0.3 | 1.08 [0.56,1.68] | 11.50 | 23.30 / 0/10 |
| tabular_q_agent | 2 | 50386 | 50386 | 46.2 | 1.32 [0.72,2.04] | 11.40 | 50.00 / 10/10 |
| tabular_q_agent | 2 | 150486 | 150486 | 142.8 | 2.12 [1.24,3.12] | 7.90 | 32.90 / 2/10 |
| tabular_q_agent | 2 | 251895 | 251895 | 239.6 | 2.08 [1.24,2.96] | 4.90 | 48.20 / 1/10 |
| tabular_q_agent | 2 | 350687 | 350687 | 357.1 | 1.12 [0.72,1.60] | 9.40 | 32.90 / 2/10 |
| dqn_agent | 2 | 0 | 0 | 0.2 | 2.56 [1.64,3.68] | 10.10 | 49.90 / 9/10 |
| dqn_agent | 2 | 50072 | 11269 | 107.1 | 3.48 [2.76,4.24] | 32.00 | 50.00 / 10/10 |
| dqn_agent | 2 | 150969 | 36493 | 373.7 | 3.96 [2.84,5.12] | 4.40 | 50.00 / 10/10 |
| dqn_agent | 2 | 250221 | 61306 | 25569.7 | 3.92 [2.76,5.24] | 4.90 | 50.00 / 10/10 |
| dqn_agent | 2 | 350310 | 86328 | 26020.9 | 4.88 [3.60,6.28] | 29.00 | 50.00 / 10/10 |

Times include driver saves/archive/probe and worker contention, exclude evaluation and process import startup. Milestones are actual chunk boundaries, not invented exact budgets.

## Final training-seed spread

| Agent | Mean combat | Min / max | Mean loot | Mean navigation | First sampled random+1 crossing (transitions per seed) |
|---|---:|---|---:|---:|---|
| tabular_q_agent | 1.813 | 1.12 / 2.16 | 13.733 | 44.300 | [250276, 250448, 150486] |
| dqn_agent | 3.520 | 2.84 / 4.88 | 12.800 | 34.633 | [50047, 50115, 0] |

Safe-random combat reference mean=1.08; crossing threshold=2.08. None means not reached at a sampled point; no interpolation.

Intervals are paired/map bootstrap,20000 resamples,RNG20260921, not variability of independent trainings. Training-seed spread is reported separately; three seeds remain a small sample. Raw per-map results are retained under each exact archive's evaluation directory. JSON links every curve point to its archived model hashes.

## Paired direct combat comparison

| Seed | DQN transitions | Table transitions | DQN minus table [map95% CI] |
|---|---:|---:|---|
| 0 | 0 | 0 | -1.08 [-1.68,-0.56] |
| 0 | 50047 | 50052 | 2.32 [1.16,3.48] |
| 0 | 150952 | 151905 | 0.04 [-0.88,0.92] |
| 0 | 250241 | 250276 | -0.44 [-2.04,1.08] |
| 0 | 350187 | 350146 | 0.68 [-0.36,1.68] |
| 1 | 0 | 0 | -1.08 [-1.68,-0.56] |
| 1 | 50115 | 50070 | 1.00 [0.16,1.84] |
| 1 | 150210 | 150172 | 1.40 [0.36,2.48] |
| 1 | 250742 | 250448 | 0.48 [-0.96,1.84] |
| 1 | 350272 | 350150 | 0.68 [-0.56,1.96] |
| 2 | 0 | 0 | 1.48 [0.44,2.64] |
| 2 | 50072 | 50386 | 2.16 [1.24,3.08] |
| 2 | 150969 | 150486 | 1.84 [0.28,3.36] |
| 2 | 250221 | 251895 | 1.84 [0.28,3.52] |
| 2 | 350310 | 350687 | 3.76 [2.36,5.24] |

Final hierarchical bootstrap resamples training seeds first, then maps within each seed (20000 draws, RNG20260921). For the direct difference each map/seed pair remains paired. Three outer units give limited evidence of training robustness:

- tabular_q_agent: 1.813 [1.120, 2.520].
- dqn_agent: 3.520 [2.547, 4.867].
- dqn_minus_table: 1.707 [0.213, 3.640].

## Visit bins at final snapshots

| Agent | Seed | Table visit bin | Decisions | Table-greedy membership | Raw score attributed |
|---|---:|---|---:|---:|---:|
| tabular_q_agent | 0 | 0 | 6233 | 1.0 | 28.0 |
| tabular_q_agent | 0 | 1-9 | 29 | 1.0 | 0.0 |
| tabular_q_agent | 0 | 10+ | 11090 | 1.0 | 724.0 |
| dqn_agent | 0 | 0 | 5200 | 1.0 | 20.0 |
| dqn_agent | 0 | 1-9 | 105 | 0.3523809523809524 | 2.0 |
| dqn_agent | 0 | 10+ | 10834 | 0.5786413143806535 | 145.0 |
| tabular_q_agent | 1 | 0 | 5627 | 1.0 | 22.0 |
| tabular_q_agent | 1 | 1-9 | 39 | 1.0 | 0.0 |
| tabular_q_agent | 1 | 10+ | 8967 | 1.0 | 652.0 |
| dqn_agent | 1 | 0 | 5820 | 1.0 | 38.0 |
| dqn_agent | 1 | 1-9 | 79 | 0.2911392405063291 | 0.0 |
| dqn_agent | 1 | 10+ | 8484 | 0.6397925506836398 | 570.0 |
| tabular_q_agent | 2 | 0 | 6149 | 1.0 | 2.0 |
| tabular_q_agent | 2 | 1-9 | 51 | 1.0 | 0.0 |
| tabular_q_agent | 2 | 10+ | 11085 | 1.0 | 449.0 |
| dqn_agent | 2 | 0 | 6447 | 1.0 | 89.0 |
| dqn_agent | 2 | 1-9 | 88 | 0.38636363636363635 | 8.0 |
| dqn_agent | 2 | 10+ | 8201 | 0.588586757712474 | 815.0 |

These are policy-dependent visited states pooled over the three validation tasks; score attribution is the next observed score increment (terminal residual includes posthumous kills). Membership accepts ties in the table greedy set. Initial unvisited ties can inflate agreement; these bins do not prove causal generalization.

## Run integrity and timing

- tabular_q_agent seed0: 350146 transitions, 350146 updates, 374.5s, 934.97 transitions/s; failure=None.
- dqn_agent seed0: 350187 transitions, 86297 updates, 936.8s, 373.80 transitions/s; failure=None.
- tabular_q_agent seed1: 350150 transitions, 350150 updates, 360.7s, 970.84 transitions/s; failure=None.
- dqn_agent seed1: 350272 transitions, 86319 updates, 889.7s, 393.68 transitions/s; failure=None.
- tabular_q_agent seed2: 350687 transitions, 350687 updates, 357.1s, 981.96 transitions/s; failure=None.
- dqn_agent seed2: 350310 transitions, 86328 updates, 26021.0s, 13.46 transitions/s; failure=None.

Overall elapsed including worker startup/evaluation: 28501.5s. Fixed probe checks and any failed seeds remain recorded. Do not equate passing |Q|<=50 with task quality.

## Actual training coverage

| Agent | Seed | Stage:scenario | Rounds | Transitions | Run max abs Q |
|---|---:|---|---:|---:|---:|
| tabular_q_agent | 0 | navigation:coin-heaven | 200 | 50052 | 18.142 |
| tabular_q_agent | 0 | crates:loot-crate | 295 | 118000 | 18.142 |
| tabular_q_agent | 0 | crates:classic | 320 | 128000 | 18.142 |
| tabular_q_agent | 0 | crates:coin-heaven | 160 | 54094 | 18.142 |
| dqn_agent | 0 | navigation:coin-heaven | 319 | 50047 | 22.474 |
| dqn_agent | 0 | crates:loot-crate | 327 | 130800 | 22.474 |
| dqn_agent | 0 | crates:coin-heaven | 180 | 31340 | 22.474 |
| dqn_agent | 0 | crates:classic | 345 | 138000 | 22.474 |
| tabular_q_agent | 1 | navigation:coin-heaven | 176 | 50070 | 21.165 |
| tabular_q_agent | 1 | crates:loot-crate | 330 | 132000 | 21.165 |
| tabular_q_agent | 1 | crates:classic | 325 | 130000 | 21.165 |
| tabular_q_agent | 1 | crates:coin-heaven | 173 | 38080 | 21.165 |
| dqn_agent | 1 | navigation:coin-heaven | 322 | 50115 | 21.946 |
| dqn_agent | 1 | crates:coin-heaven | 160 | 26957 | 21.946 |
| dqn_agent | 1 | crates:classic | 340 | 136000 | 21.946 |
| dqn_agent | 1 | crates:loot-crate | 343 | 137200 | 21.946 |
| tabular_q_agent | 2 | navigation:coin-heaven | 139 | 50386 | 17.061 |
| tabular_q_agent | 2 | crates:classic | 330 | 132000 | 17.061 |
| tabular_q_agent | 2 | crates:coin-heaven | 175 | 44701 | 17.061 |
| tabular_q_agent | 2 | crates:loot-crate | 309 | 123600 | 17.061 |
| dqn_agent | 2 | navigation:coin-heaven | 317 | 50072 | 21.356 |
| dqn_agent | 2 | crates:loot-crate | 355 | 142000 | 21.356 |
| dqn_agent | 2 | crates:classic | 310 | 124000 | 21.356 |
| dqn_agent | 2 | crates:coin-heaven | 205 | 34238 | 21.356 |

## Final available checkpoint details

Stopped runs show last available models, not completed-budget results.

| Agent | Seed | Delta ref | Kills | Coins | Suicides/25 | Survived/25 | Invalid | Timeouts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tabular_q_agent | 0 | -0.60 | 0.12 | 1.56 | 0 | 22 | 45 | 0 |
| dqn_agent | 0 | 0.08 | 0.08 | 2.44 | 1 | 18 | 10 | 0 |
| tabular_q_agent | 1 | -0.60 | 0.12 | 1.56 | 0 | 22 | 44 | 0 |
| dqn_agent | 1 | 0.08 | 0.20 | 1.84 | 1 | 22 | 19 | 0 |
| tabular_q_agent | 2 | -1.64 | 0.04 | 0.92 | 1 | 23 | 51 | 0 |
| dqn_agent | 2 | 2.12 | 0.40 | 2.88 | 0 | 23 | 18 | 0 |

All rates use all scheduled rounds, including deaths and limits. The JSON retains each map and all-round/successful-only steps separately.
