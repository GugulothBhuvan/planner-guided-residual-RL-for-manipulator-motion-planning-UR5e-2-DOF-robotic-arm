# PE9-lite: do the planar-arm design rankings transfer to the UR5e?

Success in % (first option vs second), Δ = first − second with a 95% bootstrap CI over seeds. "Clear" = that CI excludes 0 with >= 3 seeds on each trained side (a planner row is one deterministic evaluation, a fixed value). "Agree" = both arms rank the same option first (direction only).

| Design choice | 2-DOF | 2-DOF result | Clear | UR5e | UR5e result | Clear | Agree |
|---|---|---|---|---|---|---|---|
| HER agent: SAC vs TD3 | PE1 (P1) | 78.8 vs 63.6 (Δ +15.2, CI [+3.5, +29.4]) | yes | E1 (S1) | 20.7 vs 12.7 (Δ +8.0, CI [+4.0, +11.7]) | yes | yes |
| Reward: sparse+HER vs dense (curriculum on) | PE1/PE2 (P1) | 78.8 vs 5.3 (Δ +73.5, CI [+70.8, +76.0]) | yes | E2 (S2) | 4.5 vs 0.0 (Δ +4.5, CI [+3.0, +6.0]) | no | yes |
| Curriculum on vs off (SAC+HER) | PE1/PE2 (P1) | 78.8 vs 71.0 (Δ +7.8, CI [+3.1, +12.9]) | yes | S1 probe (1 seed, 120k) | 7.0 vs 0.0 (Δ +7.0, no CI) | no | yes |
| Hybrid vs base-only, middle budget | PE3 (p2_test, 6400) | 81.5 vs 84.6 (Δ -3.1, CI [-4.0, -2.1]) | yes | E5 (S4, 3000) | 71.5 vs 86.0 (Δ -14.5, CI [-17.0, -12.5]) | yes | yes |
| Hybrid vs pure SAC+HER | PE3 (p2_test, 6400) | 81.5 vs 1.1 (Δ +80.4, CI [+79.4, +81.5]) | yes | E5 (S4, 3000) | 71.5 vs 0.0 (Δ +71.5, CI [+69.0, +73.5]) | yes | yes |
| Hybrid vs base-only, middle budget | PE4 (p3_test, 2300) | 76.6 vs 85.7 (Δ -9.1, CI [-11.5, -6.6]) | yes | E5 (S4, 3000) | 71.5 vs 86.0 (Δ -14.5, CI [-17.0, -12.5]) | yes | yes |
| Hybrid vs pure SAC+HER | PE4 (p3_test, 2300) | 76.6 vs 0.2 (Δ +76.4, CI [+74.0, +78.9]) | yes | E5 (S4, 3000) | 71.5 vs 0.0 (Δ +71.5, CI [+69.0, +73.5]) | yes | yes |
| Redesign: offset vs additive residual | PE4b vs PE4 (p3_test, 2300) | 82.4 vs 76.6 (Δ +5.8, CI [+3.3, +8.4]) | yes | E5b vs E5 (S4, 3000) | 58.3 vs 71.5 (Δ -13.2, CI [-15.7, -10.7]) | yes | no |
| Redesign: offset hybrid vs base-only | PE4b (p3_test, 2300) | 82.4 vs 85.7 (Δ -3.3, CI [-3.8, -2.8]) | yes | E5b (S4, 3000) | 58.3 vs 86.0 (Δ -27.7, CI [-29.0, -26.0]) | yes | yes |

Rankings agree on 8 of 9 comparisons with results on both arms.
