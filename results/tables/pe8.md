# PE8: P5a torque reaching, exact and +-20% mass plants (300 episodes per seed)

Success in %. Trained arms: mean over seeds [95% bootstrap CI over seeds] (pooled Wilson 95%, episode-level only). Baselines: one deterministic evaluation (Wilson 95%). Final error in cm (mean). Robustness gap = success(exact) - worst perturbed success, in points.

| Method | Checkpoint | Seeds | Success, mass x1.0 | Success, mass x0.8 | Success, mass x1.2 | d_pos x1.0 | d_pos x0.8 | d_pos x1.2 | Robustness gap |
|---|---|---|---|---|---|---|---|---|---|
| B5 (computed torque) | - | - | 100.0 (98.7–100.0) | 10.3 (7.4–14.3) | 10.7 (7.7–14.7) | 0.05 | 2.10 | 2.09 | 89.7 |
| B5' (PD + gravity) | - | - | 100.0 (98.7–100.0) | 19.0 (15.0–23.8) | 15.3 (11.7–19.8) | 0.42 | 2.58 | 2.68 | 84.7 |
| residual on B5' | final | 5 | 43.0 [32.5, 52.9] (40.5–45.5) | 33.0 [27.2, 38.8] (30.7–35.4) | 30.3 [24.6, 35.1] (28.0–32.6) | 2.04 | 2.25 | 2.59 | 12.7 |
| residual on B5' | primary | 5 | 43.0 [32.5, 52.9] (40.5–45.5) | 33.0 [27.2, 38.8] (30.7–35.4) | 30.3 [24.6, 35.1] (28.0–32.6) | 2.04 | 2.25 | 2.59 | 12.7 |
| SAC+HER (torque) | best | 5 | 0.1 [0.0, 0.2] (0.0–0.4) | 0.1 [0.0, 0.2] (0.0–0.4) | 0.1 [0.0, 0.3] (0.0–0.5) | 53.06 | 52.99 | 52.66 | 0.0 |
