# PE4: P3 (narrow passage); PE4b rows included, p3_test (1000 episodes per seed)

Success and collision in %: trained arms mean over seeds [95% bootstrap CI over seeds] (pooled Wilson 95%, episode-level only); planner rows one deterministic evaluation (Wilson 95%). Gap = median L(executed) / L(best known) - 1 over successful episodes.

| Method | Budget | Checkpoint | Seeds | Success | Collision | Gap |
|---|---|---|---|---|---|---|
| B2 (full) | - | - | - | 99.9 (99.4–100.0) | 0.0 (0.0–0.4) | -0.008 |
| B3 (A*) | - | - | - | 99.5 (98.8–99.8) | 0.5 (0.2–1.2) | -0.009 |
| base-only | 700 | - | - | 80.0 (77.4–82.4) | 20.0 (17.6–22.6) | -0.010 |
| base-only | 2300 | - | - | 85.7 (83.4–87.7) | 14.2 (12.2–16.5) | -0.009 |
| base-only | 3500 | - | - | 88.4 (86.3–90.2) | 11.5 (9.7–13.6) | -0.009 |
| hybrid | 700 | primary | 5 | 71.7 [69.2, 73.9] (70.4–72.9) | 20.0 [20.0, 20.1] (19.0–21.2) | 0.000 |
| hybrid | 700 | final | 5 | 72.2 [70.4, 73.4] (70.9–73.4) | 20.1 [20.0, 20.2] (19.0–21.2) | 0.001 |
| hybrid | 2300 | primary | 5 | 76.6 [74.2, 79.1] (75.4–77.8) | 14.3 [14.2, 14.3] (13.3–15.3) | 0.001 |
| hybrid | 2300 | final | 5 | 77.5 [75.6, 78.9] (76.3–78.6) | 14.3 [14.3, 14.4] (13.4–15.3) | 0.002 |
| hybrid | 3500 | primary | 5 | 79.0 [76.6, 81.5] (77.9–80.1) | 11.7 [11.6, 11.7] (10.8–12.6) | 0.001 |
| hybrid | 3500 | final | 5 | 79.9 [78.0, 81.4] (78.8–81.0) | 11.7 [11.6, 11.7] (10.8–12.6) | 0.002 |
| pure SAC+HER | - | best | 5 | 0.2 [0.1, 0.3] (0.1–0.3) | 26.0 [23.7, 28.9] (24.8–27.3) | 0.120 |
