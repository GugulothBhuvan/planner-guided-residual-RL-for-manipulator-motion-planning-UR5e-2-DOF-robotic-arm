# PE3: P2 (static obstacles), p2_test (1000 episodes per seed)

Success and collision in %: trained arms mean over seeds [95% bootstrap CI over seeds] (pooled Wilson 95%, episode-level only); planner rows one deterministic evaluation (Wilson 95%). Gap = median L(executed) / L(best known) - 1 over successful episodes.

| Method | Budget | Checkpoint | Seeds | Success | Collision | Gap |
|---|---|---|---|---|---|---|
| B2 (full) | - | - | - | 100.0 (99.6–100.0) | 0.0 (0.0–0.4) | -0.008 |
| B3 (A*) | - | - | - | 100.0 (99.6–100.0) | 0.0 (0.0–0.4) | -0.009 |
| base-only | 4000 | - | - | 76.7 (74.0–79.2) | 23.3 (20.8–26.0) | -0.010 |
| base-only | 6400 | - | - | 84.6 (82.2–86.7) | 15.4 (13.3–17.8) | -0.009 |
| base-only | 9000 | - | - | 96.7 (95.4–97.6) | 2.3 (1.5–3.4) | -0.008 |
| hybrid | 4000 | primary | 5 | 74.0 [73.4, 74.8] (72.8–75.2) | 23.6 [23.4, 23.7] (22.4–24.8) | -0.000 |
| hybrid | 4000 | final | 5 | 74.9 [74.3, 75.5] (73.6–76.0) | 23.4 [23.4, 23.5] (22.3–24.6) | 0.000 |
| hybrid | 6400 | primary | 5 | 81.5 [80.6, 82.5] (80.4–82.6) | 15.7 [15.7, 15.8] (14.7–16.8) | 0.001 |
| hybrid | 6400 | final | 5 | 82.3 [81.8, 82.8] (81.2–83.4) | 15.7 [15.5, 15.8] (14.7–16.7) | 0.002 |
| hybrid | 9000 | primary | 5 | 92.8 [91.7, 93.8] (92.0–93.5) | 2.7 [2.7, 2.8] (2.3–3.2) | 0.004 |
| hybrid | 9000 | final | 5 | 93.7 [93.1, 94.2] (93.0–94.3) | 2.7 [2.5, 2.8] (2.3–3.2) | 0.004 |
| pure SAC+HER | - | best | 5 | 1.1 [0.7, 1.5] (0.8–1.4) | 5.0 [4.2, 5.5] (4.4–5.6) | 0.209 |
