# E5 diagnostic: residual authority around the base reference (S4 test set, primary checkpoint)

dev_max = max over steps of max_j |q_target - q_ref(t)| per episode (rad). Bound if the base cancels the residual every step: alpha * dq_max = 0.3 * 0.04 = 0.012 rad (the base action itself is clipped to +-dq_max, so the bound is exceeded only where the base saturates).

## 1. Distribution of dev_max per episode

| Arm | Budget | Seeds | Episodes | p5 | median | p95 | max | > 0.012 rad |
|---|---|---|---|---|---|---|---|---|
| base-only | 1500 | - | 200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0% |
| base-only | 3000 | - | 200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0% |
| base-only | 6000 | - | 200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0% |
| dense hybrid | 1500 | 3 | 600 | 0.0119 | 0.0268 | 0.2029 | 0.3868 | 61.0% |
| dense hybrid | 3000 | 3 | 600 | 0.0119 | 0.0313 | 0.2140 | 0.3868 | 64.0% |
| dense hybrid | 6000 | 3 | 600 | 0.0119 | 0.0352 | 0.2154 | 0.3868 | 66.3% |
| sparse hybrid | 1500 | 2 | 400 | 0.0119 | 0.0336 | 0.2616 | 0.3570 | 59.8% |
| sparse hybrid | 3000 | 2 | 400 | 0.0119 | 0.0391 | 0.2621 | 0.3570 | 63.2% |
| sparse hybrid | 6000 | 2 | 400 | 0.0119 | 0.0408 | 0.2616 | 0.3570 | 65.2% |

## 2. Final pose error on episodes where both base-only and the hybrid avoided collisions

Paired by episode (same test episode, same budget); hybrid seeds pooled. Mean and median.

| Arm | Budget | Pairs | d_pos base (cm) | d_pos hybrid (cm) | theta base (deg) | theta hybrid (deg) | success base | success hybrid |
|---|---|---|---|---|---|---|---|---|
| dense | 1500 | 471 | 0.41 / 0.41 | 0.72 / 0.66 | 0.88 / 0.74 | 1.32 / 1.24 | 100.0% | 83.9% |
| dense | 3000 | 508 | 0.41 / 0.40 | 0.72 / 0.67 | 0.89 / 0.74 | 1.31 / 1.23 | 100.0% | 84.1% |
| dense | 6000 | 538 | 1.18 / 0.41 | 1.48 / 0.67 | 2.78 / 0.74 | 3.19 / 1.26 | 98.3% | 83.1% |
| sparse | 1500 | 316 | 0.41 / 0.41 | 0.72 / 0.69 | 0.88 / 0.74 | 1.29 / 1.19 | 100.0% | 87.3% |
| sparse | 3000 | 342 | 0.41 / 0.40 | 0.72 / 0.69 | 0.89 / 0.74 | 1.30 / 1.19 | 100.0% | 86.5% |
| sparse | 6000 | 363 | 1.17 / 0.41 | 1.47 / 0.69 | 2.75 / 0.74 | 3.14 / 1.21 | 98.3% | 85.1% |

## 3. Outcome split per budget (% of episodes)

| Arm | Budget | Episodes | success | collision | timeout | pose miss |
|---|---|---|---|---|---|---|
| base-only | 1500 | 200 | 79.5% | 20.5% | 0.0% | 0.0% |
| dense hybrid | 1500 | 600 | 66.2% | 21.2% | 0.0% | 12.7% |
| sparse hybrid | 1500 | 400 | 69.2% | 20.8% | 0.0% | 10.0% |
| base-only | 3000 | 200 | 86.0% | 14.0% | 0.0% | 0.0% |
| dense hybrid | 3000 | 600 | 71.5% | 15.0% | 0.0% | 13.5% |
| sparse hybrid | 3000 | 400 | 74.2% | 14.2% | 0.0% | 11.5% |
| base-only | 6000 | 200 | 90.0% | 8.5% | 1.0% | 0.5% |
| dense hybrid | 6000 | 600 | 74.8% | 10.0% | 1.0% | 14.2% |
| sparse hybrid | 6000 | 400 | 77.5% | 9.0% | 1.0% | 12.5% |
