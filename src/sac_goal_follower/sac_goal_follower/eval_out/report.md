# SAC Evaluation Report (20 Episodes)
## Outcome summary
- Episodes: 20
- Success (goal): 18 (90.0%)
- Collision: 1 (5.0%)
- Timeout: 1 (5.0%)

## Return & efficiency
- Mean return: 14.392 (± 1.958 95% CI)
- Mean length: 118.3 steps (± 67.3 95% CI)
- Return ↔ Length corr: -0.642

## Progress dynamics
- Mean total progress (dist_start - dist_final): 5.259
- Median total progress: 5.548

## Safety (TTC + proximity)
- Mean(min TTC): 1.881 s
- Median(min TTC): 1.913 s
- Mean fraction of steps TTC<2s: 0.081
- Return ↔ minTTC corr: 0.025 (positive means safer episodes score higher)

## Control behavior
- Mean v_cmd: 0.671 m/s
- Mean |w_cmd|: 0.201 rad/s
- Return ↔ mean speed corr: 0.633
- Return ↔ mean turning corr: 0.223

## Per-episode table
| Ep | Reason | Return | Steps | MinDist | MinTTC | TTC<2s(frac) | v_mean | |w|_mean |
|---:|:------:|------:|------:|--------:|------:|------------:|-------:|---------:|
|  1 |  goal  |  17.050 |    72 |   1.446 |  2.140 |        0.000 |  0.800 |   0.236 |
|  2 | timeout |   1.019 |   765 |   3.526 |  2.191 |        0.000 |  0.043 |   0.092 |
|  3 |  goal  |  15.124 |   100 |   1.473 |  1.827 |        0.100 |  0.682 |   0.208 |
|  4 |  goal  |  16.718 |    74 |   1.459 |  2.005 |        0.000 |  0.769 |   0.235 |
|  5 |  goal  |  16.017 |    89 |   1.471 |  1.899 |        0.056 |  0.612 |   0.200 |
|  6 | collision |   2.771 |    27 |   5.154 |  1.818 |        0.333 |  0.663 |   0.197 |
|  7 |  goal  |  15.117 |   116 |   1.489 |  1.861 |        0.069 |  0.685 |   0.302 |
|  8 |  goal  |  16.817 |    77 |   1.475 |  2.222 |        0.000 |  0.700 |   0.115 |
|  9 |  goal  |  16.954 |    90 |   1.454 |  2.030 |        0.000 |  0.621 |   0.166 |
| 10 |  goal  |  14.554 |    77 |   1.458 |  1.019 |        0.182 |  0.740 |   0.235 |
| 11 |  goal  |  15.728 |    95 |   1.489 |  1.812 |        0.063 |  0.628 |   0.214 |
| 12 |  goal  |  16.604 |    74 |   1.482 |  1.926 |        0.027 |  0.804 |   0.135 |
| 13 |  goal  |  16.976 |    64 |   1.446 |  2.044 |        0.000 |  0.853 |   0.165 |
| 14 |  goal  |  12.024 |    84 |   1.476 |  1.637 |        0.333 |  0.761 |   0.264 |
| 15 |  goal  |  15.166 |    68 |   1.486 |  1.477 |        0.147 |  0.885 |   0.310 |
| 16 |  goal  |  16.898 |    92 |   1.475 |  2.125 |        0.000 |  0.595 |   0.144 |
| 17 |  goal  |  16.698 |   124 |   1.478 |  2.175 |        0.000 |  0.558 |   0.165 |
| 18 |  goal  |  13.886 |    84 |   1.459 |  1.879 |        0.226 |  0.644 |   0.167 |
| 19 |  goal  |  16.562 |   104 |   1.494 |  2.015 |        0.000 |  0.781 |   0.266 |
| 20 |  goal  |  15.144 |    90 |   1.480 |  1.513 |        0.078 |  0.601 |   0.204 |

## Interpretation notes (what these numbers usually mean)
- If success rate is low but progress_total is high, the agent moves toward the goal but fails late (often obstacle interaction / local minima).
- Many collisions with very low minTTC and high TTC<2s fraction suggests the policy is aggressive and lacks braking/avoidance margin.
- High |w_cmd| with mediocre progress suggests oscillation/zig-zag (can come from reward shaping, noisy obstacle features, or too-high entropy).
- Timeouts with decent TTC but poor progress usually means the agent is indecisive or stuck in turning behavior.
