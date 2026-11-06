# MyCobot residual diagnostic (s4_test, 200 episodes, horizon 900, final checkpoints)

Additive residual a = clip(a_base + alpha * pi_res); alpha = 0.3, dq_max = 0.5 deg per tick, so the residual moves the target by at most alpha * dq_max = **0.00262 rad** per joint per step (the bound, 1.0 below).

## 1. Size of the residual action alpha * |pi_res| per step

Mean over steps (then over episodes) and maximum over all steps, per joint J1-J6, in units of the bound (alpha * |pi_res| / alpha = |pi_res|; 1.0 = the residual output saturated).

| Seed | Budget | mean J1 | J2 | J3 | J4 | J5 | J6 | max over all steps and joints | mean (rad), all joints |
|---|---|---|---|---|---|---|---|---|---|
| s0 | 1500 | 0.832 | 0.760 | 0.378 | 0.723 | 0.711 | 0.739 | 1.000 | 0.00181 |
| s1 | 1500 | 0.725 | 0.750 | 0.739 | 0.562 | 0.739 | 0.765 | 1.000 | 0.00187 |
| s0 | 3000 | 0.830 | 0.754 | 0.373 | 0.721 | 0.707 | 0.737 | 1.000 | 0.00180 |
| s1 | 3000 | 0.724 | 0.754 | 0.734 | 0.550 | 0.737 | 0.764 | 1.000 | 0.00186 |
| s0 | 6000 | 0.827 | 0.754 | 0.369 | 0.720 | 0.709 | 0.736 | 1.000 | 0.00180 |
| s1 | 6000 | 0.725 | 0.747 | 0.733 | 0.542 | 0.738 | 0.766 | 1.000 | 0.00186 |

## 2. Deviation of the target from the reference, dev_max per episode (rad)

dev_max = max over steps of max_j |q_target - q_ref(t)|. In units of the bound in the last column.

| Arm | Budget | Episodes | p5 | median | p95 | max | median / bound |
|---|---|---|---|---|---|---|---|
| base-only | 1500 | 200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 |
| hybrid s0 | 1500 | 200 | 0.0026 | 0.0292 | 0.0798 | 0.1059 | 11.15 |
| hybrid s1 | 1500 | 200 | 0.0026 | 0.0285 | 0.0826 | 0.1344 | 10.89 |
| base-only | 3000 | 200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 |
| hybrid s0 | 3000 | 200 | 0.0026 | 0.0334 | 0.0899 | 0.1157 | 12.74 |
| hybrid s1 | 3000 | 200 | 0.0026 | 0.0292 | 0.0845 | 0.1344 | 11.17 |
| base-only | 6000 | 200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 |
| hybrid s0 | 6000 | 200 | 0.0026 | 0.0349 | 0.0899 | 0.1157 | 13.35 |
| hybrid s1 | 6000 | 200 | 0.0026 | 0.0315 | 0.0865 | 0.1344 | 12.04 |

## 3. Episodes whose outcome differs from the planner alone

Same episode, same budget. Outcome = success, collision, or other (no success, no contact).

| Seed | Budget | Success base | Success hybrid | Outcome differs | hybrid better (fail -> success) | hybrid worse (success -> fail) | success identical in the rest |
|---|---|---|---|---|---|---|---|
| s0 | 1500 | 0.895 | 0.880 | 3 / 200 | 0 | 3 | 197 |
| s1 | 1500 | 0.895 | 0.895 | 0 / 200 | 0 | 0 | 200 |
| s0 | 3000 | 0.925 | 0.915 | 2 / 200 | 0 | 2 | 198 |
| s1 | 3000 | 0.925 | 0.925 | 0 / 200 | 0 | 0 | 200 |
| s0 | 6000 | 0.935 | 0.930 | 1 / 200 | 0 | 1 | 199 |
| s1 | 6000 | 0.935 | 0.935 | 0 / 200 | 0 | 0 | 200 |

## Verdict

**The residual acts but cancels out.** The mean residual action is 0.70 of its bound (0.00183 rad per joint per step); the median dev_max is 0.0308 rad for the hybrids and 0.0000 rad for the planner alone, against a bound of 0.0026 rad. 6 of 1200 episode-runs (0.5%) end differently from the planner alone.
