# MyCobot reach (SPEC M10): learned controller against no-learning rules

100 frozen test episodes per stage (`testsets/reach_<stage>_test.npz`), deterministic policies. Success = TCP within 2 cm. Collision = any contact (table, arm, sphere). RL = DDPG + HER (two critics) unless stated; 'best' = checkpoint with the best validation success (100 validation episodes, every 10k steps), 'last' = final.

## R1 fixed start, no obstacle, ee

| Controller | Success | Collision | Final error median / max (cm) | Steps median | Training |
|---|---|---|---|---|---|
| scripted straight line | 100% | 0% | 1.91 / 2.0 | 59 | none |
| scripted detour (up over, across, down) | 46% | 0% | 2.86 / 11.0 | 200 | none |
| DDPG+HER seed 0 (best) | 27% | 0% | 3.96 / 16.3 | 200 | 100k steps, 41 min |
| DDPG+HER seed 0 (last) | 23% | 0% | 3.98 / 16.6 | 200 | 100k steps, 41 min |
| DDPG+HER seed 1 (best) | 6% | 0% | 9.54 / 16.4 | 200 | 100k steps, 42 min |
| DDPG+HER seed 1 (last) | 4% | 0% | 10.00 / 15.8 | 200 | 100k steps, 42 min |
| DDPG+HER seed 2 (best) | 68% | 0% | 1.98 / 4.4 | 78 | 100k steps, 41 min |
| DDPG+HER seed 2 (last) | 64% | 0% | 1.97 / 4.5 | 81 | 100k steps, 41 min |

RL (best checkpoints, 3 seeds pooled) 34% against the best scripted rule 100%: z = -11.5, so RL **below** it.

## R2 + sphere at the midpoint

| Controller | Success | Collision | Final error median / max (cm) | Steps median | Training |
|---|---|---|---|---|---|
| scripted straight line | 0% | 100% | 12.24 / 16.1 | 18 | none |
| scripted detour (up over, across, down) | 0% | 73% | 11.80 / 17.2 | 49 | none |
| oracle: some via-point detour found by dry runs | 0% | 0% | | | simulation look-ahead |

## R3 + random start

| Controller | Success | Collision | Final error median / max (cm) | Steps median | Training |
|---|---|---|---|---|---|
| scripted straight line | 0% | 100% | 14.31 / 20.2 | 24 | none |
| scripted detour (up over, across, down) | 0% | 38% | 14.75 / 25.4 | 250 | none |
| oracle: some via-point detour found by dry runs | 13% | 0% | | | simulation look-ahead |

## R4 R1 in joints mode

| Controller | Success | Collision | Final error median / max (cm) | Steps median | Training |
|---|---|---|---|---|---|
| scripted straight line | 100% | 0% | 1.91 / 2.0 | 59 | none |
| scripted detour (up over, across, down) | 46% | 0% | 2.86 / 11.0 | 200 | none |

