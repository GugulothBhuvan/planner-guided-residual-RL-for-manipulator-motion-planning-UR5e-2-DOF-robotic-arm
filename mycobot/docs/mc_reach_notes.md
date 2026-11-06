# MyCobot reach (SPEC M10): design choices, deviations and status

Written 2026-11-06 while the runs were in progress; the final numbers are in `results/tables/mc_reach.md` and
`results/tables/mc_residual_diagnostic.md`.

## Task 0: the residual diagnostic (done)

`scripts/mc_residual_diagnostic.py`, table `results/tables/mc_residual_diagnostic.md`. Verdict: **the residual acts but cancels out.**
The residual output averages 0.70 of its range (|pi_res| about 0.55-0.83 per joint, up to 1.0), i.e. it is not collapsed to zero, yet the
outcome differs from the planner alone in only 6 of 1200 episode-runs (seed 0: 3, 2 and 1 episodes at 1500, 3000 and 6000 checks, all
worse; seed 1: none). dev_max (largest |q_target - q_ref|) has a median of 0.03 rad against the 0.0026 rad bound, because the bound
holds only for unsaturated base steps (the base action saturates at the reference's start and corners); the planner alone has 0.

## Environment (`mcrl/envs/reach_env.py`, subclass of the pose env; the S4 env and its defaults are untouched)

- **Goal cube:** 10 cm, centre (0.12, -0.05, 0.35) m in the base frame. Chosen from `scripts/mc_reach_calibrate.py`: of a coarse grid of 48
  centres, 39 are reached 100% (60 goals each, no contact) by the scripted straight line; the chosen one reaches **500 of 500** goals with
  no contact (median 58 steps, maximum 121).
- **`ee` mode:** one damped-least-squares step on the tool-position Jacobian (damping 0.02) plus a weak null-space pull to the home joints
  (gain 0.05), scaled so no joint moves more than 0.5 deg per tick. **Deviation from the brief:** the tool orientation is not held. The
  home pose has the tool axis horizontal, holding that orientation reaches only 32% of a 5 cm grid and no 10 cm cube near home
  (`results/reach_map_home_R.pkl`), and success is position only.
- **Episode horizon:** 200 steps for R1 and R4 (the longest straight move needs 121 steps, 60%), 250 for R2 and R3 (a detour is longer).
- **Reward / termination:** sparse -1 / 0, the episode ends on success; contact (table, arm or sphere) ends it with -1/(1-gamma) = -50
  (w_c = w_s = w_m = 0), kept through the shared `SafetyHerReplayBuffer`; obstacle slots exist in every stage so the observation layout
  is the same. Verified by `tests/test_reach.py`.
- **Training:** DDPG + HER (two critics, `future`, k = 4, noise 0.1, gamma 0.98, 2 x 256, batch 256, learning rate 1e-3, 0.5 gradient steps
  per transition, 4 environments), `scripts/train_reach.py`; `n_critics` was added to the shared agent factory as an optional key (absent in
  every earlier config). About 70 steps/s per run with 3-6 runs in parallel on 12 cores: 100k steps take 25-35 minutes.

## Scripted rules on the 100 test episodes (`results/eval/reach/scripted_*.json`)

| Stage | Straight line | Detour | Oracle (some via-point detour in a dry run) |
|---|---|---|---|
| R1 fixed start, no obstacle | 100% | 46% (not a goal-directed rule without an obstacle; z_top is the start height) | |
| R2 + sphere at the midpoint | 0% (100% contact) | 0% (73% contact, the rest time out) | **0%** |
| R3 + random start | 0% (100% contact) | 0% (38% contact) | 13% |
| R4 joints mode | 100% | 46% | |

## Finding that needs a decision: R2 as specified is not solvable in `ee` mode on this arm

With the sphere on the start-goal line (+-1 cm), the home tool tip already at the workspace ceiling (z = 0.42 m, so "up over the sphere" is
above reach: the tool stops at 0.428 m) and the sphere next to link 3 (the forearm sweeps through it whenever the tool leaves home towards the
cube), **no hand-coded via-point detour (up, down, +-x, +-y; three clearances) reaches the goal without contact on any of the 100 R2 test
episodes**. The contact is with link 3, not the tool tip. An RL policy in `ee` mode can only shape the tool path, so R2 is probably out of reach
for it too. Options for the next iteration (each changes the brief, so none was applied): place the sphere off the arm's own swept volume
(for example toward the goal end of the line, or larger start-goal separation), move the cube further out, or use `joints` mode for the
obstacle stages so the elbow can move. R2 and R3 runs were started as specified so that the result is on record.

## Status at the time of writing

- **R1** (ee, fixed start, no obstacle), DDPG + HER, 3 seeds, 100k steps, 100 test episodes (best-by-validation checkpoint): **27%, 6% and 68%**,
  no contact in any run (median final error 4.0, 9.5 and 2.0 cm). **Below the 90% bar.** The scripted straight line gets 100%. Seed 2 was still
  rising (validation 30% at 60k, test 68% at the end, worst final error 4.4 cm); seeds 0 and 1 had barely started. Nothing was tuned; the brief's
  next step is to extend while rising (not yet done: `train_reach.py` does not resume, and the replay buffer is not saved) and, if a seed stays
  flat, to remove one difficulty at a time (larger `ee_step`, more exploration noise, fewer joints moving). Throughput was 40 steps/s per run with
  6 runs in parallel, 70 steps/s with 3.
- **R2** (sphere): three 100k-step runs were started as specified and were still running; their results are not in the tables yet. Given the
  oracle finding above, success near zero is expected; `scripts/eval_reach.py policy r2 ...` and `scripts/reach_report.py` collect them.
- R3, R4, SAC+HER and all hardware work (Task 2): not started. **The arm was not moved.**

## R2 sphere placement probes (2026-11-06, after "move the sphere")

Share of 40 sampled R2-like episodes solved by some via-point detour (6 directions x 3 clearances, dry runs; `scripts/reach_frac_probe.py`):

| Sphere at fraction of the start-goal line | Radius (cm) | Cube centre (m) | Start clearance | Solvable |
|---|---|---|---|---|
| 0.5 (midpoint, as specified) | 1.5-3.5 | (0.12, -0.05, 0.35) | 1 cm | 0% |
| 0.7 | 1.5-3.5 | same | 1 cm | 0% |
| 0.5 | 1.5-2.0 | same | 1 cm | 0% |
| 0.55 | 1.5-3.0 | (0.20, -0.02, 0.30), goals >= 16 cm from start | 4 cm | 5% |
| 0.55 | 1.5-3.0 | (0.16, -0.10, 0.30) | 4 cm | 0% |
| 0.55 | 1.5-3.0 | (0.18, 0.05, 0.33) | 4 cm | 5% |

Moving the sphere along the line, shrinking it, moving the cube out and keeping 4 cm from the arm at the start do not make a Cartesian detour
work. Debug runs show why: the wrist links (4, 5 and the tool body) are large and trail the tool tip by a few centimetres, so a sphere that
the tool tip passes at 7 cm is still touched by link 4 or 5, and sideways or downward via points run into the wrist before the tip moves. The
`ee` action fixes the elbow and wrist through the DLS null-space pull, so a policy cannot move them out of the way either. The three R2 policies
trained as specified end at 0% success with 55%, 88% and 59% contact on validation (100k steps).
