# SPEC — MyCobot 280 lane (real-arm companion to the UR5e study)

| | |
|---|---|
| **Status** | Draft v0.4 — 2026-11-03 |
| **Parent** | [../SPEC.md](../SPEC.md) v0.25 (UR5e). This lane reuses its agents, residual recipe and protocol. |
| **Goal** | Deploy the planner + residual pipeline on a real MyCobot 280 M5 (suction head) and test it against the same baselines. |
| **Deadline** | 24 h from 2026-11-02 (user). Scope below is cut to fit; cuts are listed in M9. |

## M1 Hardware facts (measured 2026-11-02, `results/`)

| Item | Value | Source |
|---|---|---|
| Port / library | COM13 (CH9102), 115200; `pymycobot` 4.0.7 `MyCobot280`; firmware system 7.2, basic 2.4 | `read_only_check.py` |
| Safe joint limits (deg) | J1 ±170, J2 −130..85, J3 ±155, J4 ±165, J5 ±165, J6 ±180 | guide's GUI (`hover.py`) |
| Single-call latency | `get_angles` 13 ms, `send_angles` 7 ms, `is_moving` 5 ms | `timing.json` |
| Command rule | The controller drops a command arriving < ~10 ms after a send; the library then waits 0.1 s × retries (~230 ms stall). Send-only at 25 Hz is clean. send → 30 ms → read is clean in motion: cycle ≈ 49 ms | `gap_sweep.json`, `timing_interleave.py` |
| Settled error | 0.5–0.75° short of target on every joint, in the direction of motion (deadband) | `stream_settle.json`, `track_test.json` |
| Tracking | 0.5°/tick (10°/s) at 20 Hz: clean 51 ms ticks, lag ≈ 1.5° (3 ticks). ≥ 1°/tick: comm stalls of 270–530 ms | `track_test.json` |
| FK accuracy | URDF FK + fitted tool offset (7.2, −5.3, 0.8) mm vs the arm's own `get_coords`, 15 poses: 3.6 mm 3-D RMS (2.1 per axis), 6.0 mm max | `poses.json` |

## M2 Method

Unchanged from the parent spec §3.1/§4: a cheap RRT-Connect planner with a check-count budget (Base@N, straight-line
fallback) plus an additive SAC residual (E5 recipe) on top of a reference tracker; compared with the base alone and
the full planner B2. Hybrid ≤ base is the expected outcome from the UR5e/2-DOF results; this lane tests whether it
holds on a different arm and on hardware.

## M3 Simulation model

- **Kinematics:** the official `mycobot_ros` URDF (`mycobot_280_m5`, `..._with_pump`), as `mc280/kinematics.py`. MuJoCo and numpy FK agree to 1e-13.
- **Collision:** capsules fitted to the link frames (radii 35/28/25/22/22/20 mm, suction head 15 mm), no meshes. All 15 real poses are collision-free; 23% of uniform random configurations collide (table, self). Capsule radii are estimates, not measurements (limitation).
- **Servo:** position actuators, kv/kp = 0.15 s (first-order lag matching ≈1.5° at 10°/s). The 0.5–0.75° deadband is **not** modelled in v0.1 (limitation; planned domain randomization).
- **Env:** same observation layout, action space (6-D, integrated joint target), reward and termination as the UR5e env, so UR5e actors accept it (zero-shot probe).
- **Control rate 20 Hz** (25 substeps × 2 ms). *Decision (CLAUDE.md "spec has no rule → keep comparable"):* the measured safe cycle is ~49 ms; 25 Hz would need reads every other tick.
- **Δq_max = 0.5° = 0.0087 rad** per tick (hardware limit above); `target_lead_max` 0.055 rad (UR5e 0.25 × 0.0087/0.04). V_CAP = 0.1745 rad/s; B2 time scaling V_MAX = 0.92 V_CAP = 0.16 rad/s (UR5e ratio), A_MAX = 0.5 rad/s².
- **Success** 1 cm / 5° (unchanged). Hardware repeatability (≈0.7° per joint ≈ 3–5 mm at the tool) is within it but not by a wide margin.
- **IK:** batched damped least squares from 48 seeds (q_start + goal-hash-seeded random), 25 ms per goal; never uses the episode's q_goal.
- **Obstacles:** spheres with radius 15–35 mm (UR5e 40–100 mm scaled by the reach ratio), placement ±5 cm around the start–goal line, 1–4 per episode. `d_safe` 0.02 m, base inflation 0.017 m.

## M4 Task and test sets (S4 only)

- Goals sampled in joint space within **0.9 rad per joint** of the start (UR5e: whole joint box). *Decision:* at 10°/s the whole box means 20–60 s motions; the bound keeps B2 durations practical and the horizon trainable. Deviation recorded; applies to train, validation and test alike.
- B2 must finish within **20 s** (episodes B2 cannot are rejected, like episodes without any path). Horizon rule unchanged: max(200, ⌈1.25 × p99 B2 duration / 0.05 s⌉), written to `configs/env/s4.yaml` by `make_testsets.py`.
- Test: 200 episodes (seeds from 3 200 000); validation: 100 (1 200 000). Frozen in `testsets/`.
- **Horizon changed to 900 steps (45 s), decision 2026-11-02 (user).** The 489-step sets exposed that the cheap planner's reference paths (20 shortcuts) are ~2× longer than B2's (p50 294, p90 754, max 1302 steps on the first 100 test episodes): all 22 collision-free failures of base@6000 in that sample had a reference longer than 489, so the base was time-limited, not safety-limited (UR5e: safety-limited). The episodes are unchanged; only the horizon is (`scripts/set_horizon.py`). The first 1M-step runs (489) were stopped at ~40k steps and archived with their evaluations in `results/archive_h489/`; frozen sets at 489 in `testsets/archive_h489/`. Everything reported below this line is at 900 unless marked.
- **Built 2026-11-02 (at 489):** test 200 episodes (71 seeds rejected), B2 success 100%, B2 duration p50 10.0 s / p99 19.6 s / max 19.8 s, B2 planning time p50 0.09 s; validation 100 (45 rejected), p50 9.5 s / p99 18.6 s. **S4 horizon = 489 steps (24.5 s)** (rule above). Rejection rate 26% means the test set is the subset of draws that B2 completes within 20 s; this is stated wherever its numbers are reported.
- E1/E2 (reaching without obstacles) and the OOD set are **not** repeated on this arm.

## M5 Training

- Residual SAC, dense reward, no HER, 4 envs, 0.25 gradient steps/transition, 3×256, base budgets {1500, 3000, 6000} per episode: the E5 recipe.
- **Schedule** = the UR5e's: 1 000 000 steps, critic warm-up 50 000 gradient steps, α ramp 150 000 steps; seeds 0, 1 (the UR5e had 3). Horizon 900.
- Gate (standing rule): steps/s measured and reported before launching.

## M5b Results so far (horizon 900, `s4_test`, 200 episodes, success / collision %, `results/eval/s4_test/`)

| Method | 1500 checks | 3000 checks | 6000 checks |
|---|---|---|---|
| B2 (full planner) | 100 / 0.0 | 100 / 0.0 | 100 / 0.0 |
| Cheap planner alone | 89.5 / 10.0 | 92.5 / 5.0 | 93.5 / 2.5 |
| UR5e E5 residual, zero-shot (primary ckpt, seed 0) | 89.5 / 10.0 | 92.0 / 5.5 | 93.0 / 3.0 |
| MyCobot-trained residual, seed 0 (final, 1M) | 88.0 / 11.5 | 91.5 / 6.0 | 93.0 / 3.0 |
| MyCobot-trained residual, seed 1 (final, 1M) | 89.5 / 10.0 | 92.5 / 5.0 | 93.5 / 2.5 |

At horizon 489 (archived) the cheap planner was flat at 84.0 / 81.0 / 80.5: time-limited. Training speed with other jobs running: ~69 steps/s per run (50k steps in 12 min); validation = planner alone (0.93) while the actor is frozen.
M5 finished 2026-11-03 06:30 (both seeds, no restarts after the 04:21 resume). Validation ended at 0.92 for both seeds (planner alone 0.93). Test: the retrained residual equals the cheap planner within the 95% intervals (e.g. 1500 checks: 88.0 [83.5, 92.5] and 89.5 [85.0, 93.5] vs 89.5); seed 1 matches it exactly, seed 0 is 1–1.5 points lower. Same finding as the UR5e study: the residual does not beat its planner; retraining on the MyCobot sim does not change it.
Dry-run of the runner with a simulated arm (lag 0.15 s, deadband 0.7°, noise 0.1°), 30 episodes: B2 30/30, base@3000 27/30 (2 collisions); median final error 0.7 cm / 3.3° against the 1 cm / 5° limits, so the deadband uses most of the margin.

## M6 Evaluation

`mycobot/scripts/evaluate.py`: b2, base@N, hybrid@N on `s4_test` at N ∈ {1500, 3000, 6000}; **zero-shot probe** = the UR5e E5 residual actor (primary checkpoint) run unchanged in the MyCobot env at the same budgets; trained MyCobot hybrid; success/collision with 95% intervals as in the parent spec §9.

## M7 Real arm

- Runner on the 20 Hz send → 30 ms → read cycle; all moves through the same integrated-target interface; joint limits, speed cap (setting 20) and `stop()` on any error or tracking deviation > 5°.
- Obstacles are **virtual keep-out spheres** (checked on the commanded configuration; the arm never touches anything).
- Scenes: the first N episodes of `s4_test`, same order for every controller; metrics: success against the arm's own `get_coords` (tool position/orientation), virtual collisions, time.

### M7b Real-arm results (2026-11-03, `results/real/`; first version in `results/real/v1_no_trim/`; runner `scripts/real_batch.sh`)

First 10 `s4_test` episodes, speed 20, virtual obstacles, one run per method and episode; hybrid = seed-1 final residual, cheap planner at 3000 checks. Simulation, same 10 episodes: 10/10 for every method.

| Method | v1 (no correction): success | final error median (max) | v2 (final-approach correction `TrimArm`): success | final error median (max) | orientation v2 median (max) |
|---|---|---|---|---|---|
| B2 | 9 / 10 | 0.79 (1.46) cm | **10 / 10** | 0.69 (0.96) cm | 3.5 (4.3) deg |
| Cheap planner @3000 | 10 / 10 | 0.82 (0.97) cm | **10 / 10** | 0.59 (0.89) cm | 3.2 (4.2) deg |
| Hybrid @3000 | 10 / 10 | 0.67 (0.97) cm | **10 / 10** | 0.53 (0.88) cm | 2.8 (4.2) deg |

No collisions and no safety aborts in any of the 60 runs (30 per version).

- v1 failure: B2 episode 9 stopped 1.46 cm from the goal and stayed there to the horizon; a re-run was identical (same encoder counts), so systematic. The target was static and the arm stood 1.84 deg short on J2 (shoulder, gravity-loaded; others 0.6-1.05 deg), more than the 0.5-0.75 deg deadband measured unloaded.
- Fix (decision, M8): `TrimArm` (mcrl/arm.py): while the nominal target holds still (8 ticks, < 0.15 deg/tick), nudge the commanded target by the measured shortfall (gain 0.2, <= 0.1 deg/tick, |trim| <= 2 deg per joint); it decays to zero while the target moves, so paths are unchanged. Same for all methods; the 1 deg step and 5 deg lag checks apply to the sum. All 30 runs were repeated with it; `--no-trim` reproduces v1.
- With the correction every final error is below 1 cm (max 0.96) and 4.3 deg; the margin is still thin. The correction is a hardware workaround, not part of the trained method.
- Hybrid and cheap planner are indistinguishable at n = 10 (and in simulation): the residual does not beat its planner on the real arm. Ten episodes cannot separate differences below roughly 25 points.
- The two versions were not run in randomised order (v1 first, v2 later, same arm), so the improvement could include drift; it is large only for the one stalled episode.
- Operator remark: slight vibration during motion. Likely causes (not tested): a new 50 ms command each tick (firmware restarts its motion profile), the servo deadband, gravity load on J2, encoder quantisation (0.088 deg; roughly 60% of the measured roughness by estimate). Not quantified.
- Obstacle positions are given, not sensed (no camera). Real-object detection is future work.

## M8 Decisions log

| Date | Decision | Why |
|---|---|---|
| 2026-11-02 | 20 Hz, Δq_max 0.5°/tick | measured safe comms cycle and tracking |
| 2026-11-02 | goal distance ≤ 0.9 rad; B2 ≤ 20 s | practical durations at 10°/s; trainable horizon |
| 2026-11-02 | batched DLS IK | scalar version took ~1 s and starved B2's 2 s budget (B2 success 17–25% → 97%) |
| 2026-11-02 | slimmer capsules (20–35 mm) | first guess (25–40 mm) made B2 fail on clutter the real arm can pass |
| 2026-11-02 | 1M-step residual runs × 2 seeds, full UR5e schedule (warm-up 50k, α ramp 150k) | throughput 116 steps/s measured; user chose 1M |
| 2026-11-02 | horizon 489 → 900 (user), runs restarted | cheap-planner paths outlast 489 (timeouts, not collisions) |
| 2026-11-03 | `TrimArm` final-approach correction on the real arm, all 30 hardware runs repeated (v1 archived) | J2 stopped 1.84 deg short under load (B2 ep 9, reproducible); user approved; same controller for all methods keeps them comparable |

## M9 Cuts and limitations

- No E1/E2/OOD, no E5b (offset) on this arm, no PE-style planar study here.
- 400k steps and 2 seeds: results are indicative, not comparable one-to-one with the 1M × 3 UR5e runs.
- Capsule geometry and tool length are estimates; servo deadband unmodelled; hardware evaluation on a small number of scenes.
