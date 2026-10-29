# SPEC — RL Trajectory & Motion Planning for a Planar 2-DOF Arm

| | |
|---|---|
| **Project** | MTP companion study — planar 2-DOF arm (lives in `arm2dof/` beside the UR5e project) |
| **Author** | Bhuvan Raj, IIT Kharagpur |
| **Status** | Draft v0.12 — 2026-10-30. v0.12: PE9-lite "clear" counts planner rows as fixed values (§8). v0.11: PE4b uses the E5b v2 offset residual; v1 runs archived (§8). v0.10: PE4b recorded before training (accumulated residual on P3) and added to PE9-lite (§8). v0.9: PE3/PE4 residual warm-up and α ramp scaled to the 150k-step run (§7.4); P5a test set frozen and the PE8 evaluation protocol (§9.2); residual checkpoint rule and table intervals as on the UR5e; PE9-lite defined (§8); seeds 2–4 on Kaggle (§7.1). v0.8: PE8 base-only (B5') reference on the training validation episodes (§8); 2-DOF lane paused entirely while UR5e E5 runs (§11). v0.7: absorbing collision penalty on the planar arm too (UR5e §6.7); PE3 v1 archived as `pe3_v1_crash_exploit`; frozen P1–P3 test/validation sets and `2-dof/scripts/evaluate.py`; P1 horizon vs quintic baselines (§4). v0.6: PE3/PE4 at 5 seeds and the 2-DOF lane at 1 worker while UR5e E5 runs (§7.1, §11); PE1 v2 results incl. PPO 0% (§8). v0.5: P2/P3 horizons 339/303 from B2 durations (§4); scene connectivity enforced in the env and a P2 layout that cannot split the C-space (§6.4); full B2 tries every IK branch with a 5 s cap (§7.2); residual base budgets calibrated to the UR5e base plan rates and drawn per episode (§7.4); P2/P3 mirror UR5e S4/E5 (dense default, no curriculum); PE2–PE4 configured and chained (§8); two-panel viewer built (§10). v0.4: q₁ wraps in P1–P3 and the observation drops q₁/q_limit (§5.1, §6.2, §6.3); P1 horizon 80 steps (§4); P1 uses the UR5e goal-distance curriculum with a fixed 1 cm tolerance (§6.5); PE1 v1 archived as `pe1_aborted_nowrap`, rerun as `pe1_v2` (§8). v0.3: dynamics in scope; own thesis chapter; 1-week build; residual/HER fixes shared with UR5e v0.3 |
| **Parent spec** | `../SPEC.md` (UR5e) |

> Numeric values marked **(default)** are starting points to tune and record in configs.

---

## 1. Purpose & Role in the Thesis

The 2-DOF planar arm is a **fast, fully interpretable testbed** for validating the methodology used on the UR5e. It plays four roles:

1. **De-risking.** Algorithms, reward terms, HER relabeling, curriculum and residual RL get debugged here in minutes before the same designs are moved to the UR5e.
2. **Exact analysis.** The configuration space (C-space) is 2D, so it can be computed exactly on a grid. That allows near-optimal reference paths and a **gap to the best-known path** (§9.1), which is impossible to compute for 6-DOF.
3. **Interpretability.** Policies, value functions and trajectories can be plotted directly in C-space alongside the workspace, giving strong thesis figures.
4. **Dynamics.** The 2-link equations of motion are small enough to derive, verify and control directly, so this study also covers **torque-level control**, which the UR5e work does not.

**Thesis placement (decided):** this study is **its own chapter**, framed as **methodology validation**. Its job is to validate the methods and design choices that the UR5e chapter relies on. Proposed chapter outline:

1. Planar arm model: kinematics and dynamics (§5).
2. Environment and baselines, including the exact C-space planner (§6–7).
3. Kinematic planning results: algorithms, gap to best-known path, residual RL (PE1–PE6).
4. Torque-control results: RL vs computed-torque control under model error (PE8).
5. Design recommendations carried to the UR5e and whether they transferred (PE9).

**Kinematic limitation, stated up front:** a planar 2-DOF arm controls only end-effector **position (x, y)**. Its orientation, q₁ + q₂, is determined by the position up to the elbow-up/elbow-down choice. Goals are therefore position-only. Orientation-related questions (RQ2 in the UR5e spec) are out of scope here.

## 2. Goals / Non-Goals

**Goals**

- G1. A goal-conditioned planar 2-DOF environment (Gymnasium API) for reaching and obstacle-aware motion planning.
- G2. A benchmark of DDPG/TD3/SAC (+HER) and PPO, with many seeds, which is cheap here.
- G3. Exact C-space baselines (grid A*) and a gap-to-best-known-path analysis of the RL trajectories.
- G4. Residual RL over classical planners, validated here before it is used on the UR5e.
- G5. A two-panel interactive viewer (workspace + C-space) with draggable goal and obstacles.
- G6. **Torque control with full 2-link dynamics**, compared against computed-torque control under exact and perturbed models.

**Non-Goals**

- Orientation goals, 3D motion, perception, real hardware.

## 3. Research Questions

| ID | Question | Primary metric |
|---|---|---|
| P-RQ1 | Which algorithm is most sample-efficient and stable for 2-DOF reaching / planning? | Success vs steps (10 seeds) |
| P-RQ2 | How far from optimal are RL trajectories in cluttered C-space? | Gap to best-known path (§9.1) |
| P-RQ3 | Does residual RL beat pure RL and pure classical planning in clutter? | Success, collision, jerk, latency |
| P-RQ4 | Does the learned value function / policy field reflect C-space obstacle structure? | Qualitative C-space maps; value–distance correlation |
| P-RQ5 | Which design choices from 2-DOF carry over to UR5e? | Ranking agreement between 2-DOF and UR5e results |
| P-RQ6 | Can an RL torque policy match computed-torque control, and is it more robust under model error? | Success, tracking error, energy under ±20% mass error |

## 4. Task Ladder

| Stage | Task | Obstacles | Episode length (default) | Target | Week 1? |
|---|---|---|---|---|---|
| P0 | Gymnasium `Reacher-v5` sanity run (SAC/TD3/PPO) | — | 50 | Matches known learning curves | Optional |
| P1 | Position reaching (kinematic) | — | 80 steps (3.2 s) | ≥98% success at 1 cm | Yes |
| P2 | Reaching (kinematic) | 2 static circles, fixed layout | **339** | ≥95% success, ≤2% collision | Yes |
| P3 | Reaching (kinematic) | 1–K circles, randomized | **303** | Within 15% of A* path length at ≥90% success | Yes |
| P4 | Reaching (kinematic) | Moving circles (≤0.2 m/s) | 120 | Collision rate below the replanning baseline | Phase 2 |
| P5a | Reaching with **torque control** and gravity | — | **307** | ≥90% success; robust to ±20% mass error | Yes |
| P5b | Torque control + static obstacles | 1–2 static circles | 120 | ≥85% success, ≤5% collision | Stretch |

As in the UR5e spec, stages are **time-boxed, not gated**: misses are reported, not blocking.

**P1 horizon rule (2026-10-27):** 80 steps = 1.25 × 63, the steps the farthest P1 goal needs at Δq_max = 0.05 rad/step with q₁ wrapping and the best IK branch (2000 sampled episodes; 100% reachable by 63 steps). Same 1.25× margin as the UR5e S4 rule. The original 50 steps reached only 68.8% of goals with q₁ clipped (§6.2).
- **Finding (2026-10-28): the 80-step P1 horizon is too short for quintic baselines.** It was sized from moves at the env's top joint speed (Δq_max per step), but a quintic's peak speed is 1.875× its average, so the B2/B3 quintic references on the P1 test set need median 80.5, p90 123 and max 144 steps (50% exceed 80). At 80 steps B2/B3 reach only ~40–50% of P1 goals, all without collision. **Decision:** the horizon stays at 80 (PE1/PE2 trained with it; the RL agents use full speed), and P1 classical baselines are reported twice: at the 80-step horizon (same budget as RL) and with their own duration (unbounded), so the comparison is not read as "planners fail on P1". **P2/P3 horizon rule (2026-10-27), as UR5e S4:** T = max(80, ⌈1.25 × p99 full-B2 duration⌉) in control steps, from 1000 sampled episodes per stage (`2-dof/scripts/horizon_check.py`, results in `configs/env/horizon_check.json`). Full B2 solved all 2000 episodes (after the §6.4 and §7.2 fixes). P2: median 86, p99 271, max 318 steps → **339**. P3: median 81, p99 242, max 326 → **303**. They are far above v0.3's 100 because feasible detours often run the long way round the q₁ circle and the quintic stops at every waypoint.
- **P2/P3 mirror UR5e S4 (decision 2026-10-27):** dense reward by default (the pure-RL run overrides sparse + HER, as E5 does) and no goal-distance curriculum (E5 runs none), so PE3/PE4 are the planar analogue of E5.

## 5. Arm Model

### 5.1 Parameters

| Parameter | Value (default) |
|---|---|
| Link lengths | l₁ = 0.5 m, l₂ = 0.4 m |
| Reachable workspace | Annulus, 0.1 m ≤ r ≤ 0.9 m |
| Joint limits | q₁ **continuous** (wraps at ±π, no limit) in P1–P3; q₂ ∈ [−0.9π, 0.9π] (prevents full fold-back self-overlap) |
| Link collision shape | Capsules, radius 0.03 m |
| Base | Fixed at origin |
| Plane | Horizontal for P1–P4 (gravity irrelevant); **vertical for P5** (gravity acts) |
| Masses / inertia (P5) | Uniform rods: m₁ = m₂ = 1 kg, COM at l/2, I_i = m_i l_i² / 12 about COM |
| Joint damping (P5) | b₁ = b₂ = 0.05 N·m·s/rad **(default)** |
| Torque limits (P5) | τ₁ ≤ 20 N·m, τ₂ ≤ 10 N·m (≥2× worst-case static gravity load: ≈9.3 and ≈2.0 N·m) |

### 5.2 Kinematics

- **Forward kinematics:** `x = l₁cos q₁ + l₂cos(q₁+q₂)`, `y = l₁sin q₁ + l₂sin(q₁+q₂)`.
- **Inverse kinematics:** closed form with 2 solutions (elbow-up / elbow-down). Unreachable goals are rejected at sampling time.
- **Jacobian:** analytic 2×2; singular when q₂ = 0 or ±π (fully stretched or folded).

### 5.3 Dynamics (P5)

Equations of motion: `M(q) q̈ + C(q, q̇) q̇ + g(q) + B q̇ = τ`, with `lc_i = l_i / 2` and `h = m₂ l₁ lc₂ sin q₂`:

```
M₁₁ = m₁lc₁² + I₁ + m₂(l₁² + lc₂² + 2 l₁ lc₂ cos q₂) + I₂
M₁₂ = M₂₁ = m₂(lc₂² + l₁ lc₂ cos q₂) + I₂
M₂₂ = m₂lc₂² + I₂

C(q, q̇) = [ −h q̇₂   −h (q̇₁ + q̇₂) ]
          [  h q̇₁         0       ]

g(q) = [ (m₁lc₁ + m₂l₁) g cos q₁ + m₂lc₂ g cos(q₁+q₂) ]
       [  m₂lc₂ g cos(q₁+q₂)                          ]
```

(q measured from the horizontal; g = 9.81 m/s².)

- **Simulation:** MuJoCo integrates the dynamics. The analytic model above is used by the computed-torque baseline and is verified against MuJoCo (§13).
- **Gravity must act on the P5 arm.** The UR5e scene uses actuator gravity compensation as a modeling assumption for its position-controlled arm (UR5e spec §6.1); the P5 torque model must not. Its XML (`arm2dof.xml`, torque variant) is written separately and sets no `gravcomp` on bodies and no `actuatorgravcomp` on joints, and a test enforces this (§13). Baseline B5' compensates gravity in its *controller* (`ĝ(q)` term), never in the model. The P1 MuJoCo cross-check runs in the horizontal plane (§5.1), so gravity compensation does not arise there.
- **Property checks:** M is symmetric positive definite, and `Ṁ − 2C` is skew-symmetric. Both are unit-tested; they catch sign and index errors in the derivation.
- **Built 2026-10-28:** `assets/arm2dof_torque.xml` (vertical x–z plane, hinge axis −y so q is measured from +x toward +z, RK4, links do not collide), `arm2dof/dynamics/model.py` (M, C, g, damping, forward/inverse dynamics, energy; `Params.scaled()` for the ±20% plants). Tested: M SPD, Ṁ − 2C skew, analytic M = MuJoCo's to 1e-9 and q̈ to 1e-6, gravity acts (no gravcomp; from rest at q₁ = 0 the arm falls with q̈ = −M⁻¹g), energy conserved within 1e-3 over 10 s without torque or damping.
- **P5a env (`envs/torque_env.py`):** 125 Hz, τ = a·τ_max, starts at rest. Success needs the arm to stop, so the speed lives in the goal space to keep HER exact: achieved_goal = [x, z, ‖q̇‖/10], desired_goal = [x_g, z_g, 0], success = position within ε and achieved speed < 0.1 rad/s. Observation (11): sin/cos q, q₂/q₂_lim, q̇/10, EE, previous action. Perturbed plants via `mass_factor` (masses and inertias scaled).
- **B5 / B5' (`control/controllers.py`):** both track a nearest-IK-branch quintic with v ≤ 3 rad/s, a ≤ 8 rad/s², limits chosen so the nominal feedforward torque stays ≤ 90% of τ_max on every reference (tested). B5: K_p 100, K_d 20 as specified. **B5' gains (decision 2026-10-28; the spec gave none):** the same K_p, K_d scaled by the diagonal of the nominal M at q₂ = 0, so both controllers have the same nominal per-joint stiffness. Both reach and stop on 100% of 300 sampled episodes with the nominal plant.
- **P5a horizon (§4 rule, 2026-10-28):** reference durations over 300 episodes: median 143.5, p99 245, max 268 steps at 125 Hz → T = ⌈1.25 × 245⌉ = **307 steps (2.46 s)**; v0.3's 100 steps (0.8 s) could not fit a torque-feasible move of more than ~1 rad. B5 settles (and stays) by p99 232 steps.

## 6. Environment Specification

### 6.1 Simulator Choice (decision)

- **P1–P4: pure NumPy kinematic environment.**
  - The policy commands joint-position increments, so no dynamics are needed. This matches the UR5e control interface.
  - Collision checks are exact and analytic: capsule (segment) vs circle distance in closed form.
  - Expected throughput is >100k steps/s on CPU. Ten-seed sweeps run in minutes.
- **P5: MuJoCo** 2-link model (custom XML, hinge joints, motor actuators, damping) for real dynamics.
- A MuJoCo version of P1 (position actuators) is kept as a **cross-check**: the same policy should give the same results to within noise. This guards against kinematic-env bugs. It uses the same **integrated-target rule as the UR5e** (§6.2), because re-anchoring targets on the measured q under a damped position servo caps the joint speed far below Δq_max/dt (measured on the UR5e: ~0.2 rad/s instead of 1 rad/s).

### 6.2 Control

- **P1–P4:** `a ∈ [−1,1]²`, with `q ← clip(q + a·Δq_max, limits)` and `Δq_max = 0.05 rad/step` **(default)**. Control runs at 25 Hz (dt = 0.04 s).
- **q₁ wraps at ±π** (config `wrap_q1: true` in every stage config, P1–P3), matching the cylinder C-space of §7.2: the planners and A* cross the seam, so the env must too. The flag's code default stays off only so the archived PE1 v1 config still describes what ran.
  - **Finding (2026-10-27, resolved):** with q₁ clipped, only 68.8% of P1 goals were reachable within the 50-step horizon even with the best IK branch (2000 sampled episodes; p90 73 steps, max 119), so PE1 v1 was capped near 69% success. With q₁ wrapping: 84% at 50 steps, 100% by 63 steps. **Decision:** PE1 v1 stopped and archived as `pe1_aborted_nowrap`; P1–P3 wrap; P1 horizon 80 steps (§4); PE1 rerun as `pe1_v2`.
  - All joint differences touching q₁ are taken the short way round: the q̇ observation, the goal-distance curriculum, trackers and metrics.
- **P1 MuJoCo cross-check (position actuators):** integrated targets, as on the UR5e.
  - `q_target ← clip(q_target + a·Δq_max, limits)`, then `q_target ← q + clip(q_target − q, −0.25, 0.25)` (rad).
  - `q_target = q` at reset.
  - In the kinematic env `q_target ≡ q` (the arm is the target), so both envs share one action rule and one observation layout (§6.3).
- **P5:** `a ∈ [−1,1]²` maps to τ = a · τ_max. MuJoCo timestep 0.002 s; frame skip 4, giving **125 Hz** control **(default)**. Torque control needs a faster loop than position increments to stay stable against gravity.

### 6.3 Observation (dict, Gymnasium GoalEnv)

| Key | Content | Dim |
|---|---|---|
| `observation` | sin q, cos q | 4 |
| | q / q_limit for **limited joints only**: q₂/0.9π (q₁/π too only if q₁ is clipped). A wrapping q₁ is continuous on the circle but q₁/π jumps from +1 to −1 at the seam, so it is left out; sin/cos q₁ carry it | 1 |
| | q̇ in units of Δq_max/step (wrap-aware finite difference in P1–P4, true velocity in P5) | 2 |
| | EE position | 2 |
| | previous action | 2 |
| | (q_target − q) / 0.25 (always 0 in the kinematic env; kept so one policy runs on both P1 envs) | 2 |
| | *(P2+, P5b)* obstacle slots: K × [center 2, radius 1, mask 1], K=4 | 16 |
| | *(P2+, P5b)* per-link min distance to obstacles, clipped at 0.3 m | 2 |
| | *(P4)* obstacle velocities, K × 2 | 8 |
| | *(residual)* q_ref(t), q_ref(t+5) | 4 |
| `achieved_goal` | EE (x, y) | 2 |
| `desired_goal` | target (x, y) | 2 |

**`observation` is goal-independent** (UR5e spec §6.4). HER relabels `desired_goal` only, so a goal-derived feature in `observation` (v0.3 had "goal − EE") would describe the wrong goal on relabeled samples. The policy computes `desired − achieved` itself with the shared `GoalErrorExtractor`. This bug cost the UR5e E1 its first launch.

### 6.4 Sampling

- **Goals:** sample q uniformly within limits, then goal = FK(q). Reject the sample if any link collides. This guarantees reachable, collision-free goals.
- **Start:** independent random q, with the same rejection rule. In P5, starts are at rest (q̇ = 0).
- **Obstacles (P3):** circles with radius ∈ [0.05, 0.15] m in the workspace annulus. Reject a scene if it intersects the start or goal configuration, or if the start and goal lie in **disconnected C-space components** (checked exactly by flood fill on the C-space grid, §7.3).
  - **Enforced in the env at every reset, not only when test sets are built (2026-10-27):** a scene is accepted when some collision-free IK branch of the goal shares the start's free component (`require_connected`, exact 1° grid, q₁-wrapped flood fill; ~0.17 s per reset, cached for the fixed P2 layout). Without it, 22% of P3 and 48% of P2 training episodes had unreachable goals. Tested.
- **P2 layout (2026-10-27):** circles at (0.65, 0.30) r 0.10 and (−0.45, −0.60) r 0.12. Both satisfy ρ − r ≥ 0.6 m, outside link 1's reach (0.53 m incl. capsule radius), so they block link 2 only. The first layout (copied from a unit test) had both circles inside link 1's reach; each became a full-height band in C-space and the two bands split it in half.

### 6.5 Success, Rewards, Termination

- **Success (P1–P4):** `‖p_goal − p_ee‖ < ε`, ε = 1 cm, fixed.
- **Goal-distance curriculum (training, P1–P3), same rule as UR5e E1 (UR5e spec §6.5):** goal configurations lie within a per-joint bound of the start, max_j |q_goal,j − q_start,j| ≤ level (q₁ wrap-aware), starting at 0.6 rad and raised by 0.3 rad whenever rolling success over the last 100 current-level episodes exceeds 80%, up to the full range. 20% of goals are always full-range; validation uses the full distribution. Decision 2026-10-27: this replaces v0.3's tolerance curriculum (ε 5 cm → 1 cm) so the 2-DOF matches E1 for PE9; two curricula driven by one rolling-success signal would also interfere. The tolerance curriculum stays available (`tol_curriculum`) as a PE2 ablation.
  - Budget note: 50k steps at 80 steps/episode is ~625 episodes, so the level can rise at most ~6 times (to ≤2.4 rad) in PE1. The full-range 20% and full-distribution validation keep the reported number honest.
- **Success (P5):** the same position condition **and** `‖q̇‖ < 0.1 rad/s`, so the arm has to stop at the goal, not swing through it.
- **Sparse (HER):** r = 0 on success, −1 otherwise, plus the goal-independent terms below.
- **Dense:**

  ```
  r = − w_p·d − w_a·‖a‖² − w_s·‖a_t − a_{t−1}‖²
      − w_m·Σ_links max(0, d_safe − d_link) − w_c·1[collision] + R_succ·1[success]
  ```

  Defaults: `w_p=1.0, w_a=0.01, w_s=0.05, w_m=5.0, d_safe=0.05 m, w_c=10, R_succ=5`. In P5, `‖a‖²` is a normalized torque (energy) penalty.
- **HER reward decomposition:** collision, margin and smoothness/energy penalties are stored in `info` and re-added inside `compute_reward`. This is the same contract as UR5e spec §6.7 and is covered by the same unit test.
- **Termination:**
  - collision (P2+);
  - dense mode only: success held for 5 steps.
  - Truncation happens at the episode length.

## 7. Methods

### 7.1 RL Agents

SB3 DDPG / TD3 / SAC (+HER) and PPO, with the same defaults as UR5e spec §7.1, **including the network [256, 256, 256], 0.25 gradient steps per transition and batch 256**. Compute is cheap here, but matching the UR5e settings keeps the design-transfer comparison (PE9) clean.

- **Update ratio is per transition:** SB3 counts `train_freq` in vec-env steps, so `gradient_steps = 0.25 × n_envs` with `train_freq=1` (4 envs → 1 gradient step per vec step). Use the UR5e factory (`agents/factory.py`), which does this conversion.
- **HER buffer:** the UR5e `SafetyHerReplayBuffer`, so the goal-independent penalties survive relabeling.

Exceptions:

- **PPO environments:** 32 parallel environments, which cost almost nothing here.
- **Seeds:** 10 per configuration for kinematic stages; 5 for P5 (MuJoCo is slower). **Exception (decision 2026-10-27):** PE3/PE4 run 5 seeds (the minimum) while UR5e E5 has the laptop; seeds 5–9 are added later only if time remains. **Where (decision 2026-10-29):** PE3, PE4 and PE8 seeds 2–4 run on Kaggle (`notebooks/colab_pe_s234.ipynb`, parallel jobs at 1 torch thread), seeds 0–1 on the laptop (UR5e SPEC §11).

### 7.2 Classical Baselines

- **B1 — IK + quintic** (P1): choose the IK branch nearest to the current q, then joint-space quintic interpolation with velocity/acceleration-limited duration.
- **B2 — RRT-Connect + shortcut smoothing + quintic time scaling** (P2–P4). Tries every collision-free IK branch of the goal, nearest first, each with an equal share of the cap (as the UR5e B2, which tries all branches). **Cap 5 s (decision 2026-10-27; v0.3 said 0.5 s):** at 0.5 s the Python RRT (edge resolution 0.005 rad) missed ~25% of feasible P2 episodes under load, and every one of them was solved in 0.65–3.6 s. B2 is the expensive reference, as on the UR5e (2 s there), so it must solve essentially every feasible episode; its latency is reported anyway.
- **B3 — Grid A\*** (reference path, P2–P3):
  - Exact C-space occupancy grid at 1° resolution (360 × 324 cells within limits), computed with analytic collision checks.
  - 8-connected A* minimizes joint-space path length **on the grid metric**, which overestimates the true joint-space length by up to ~8% (diagonal/axis moves only). The A* reference is therefore the A* path followed by any-angle shortcut smoothing (2000 iterations); the **best-known path** per scene is the shorter of this and RRT-Connect + shortcut (§9.1); a raw grid path is not used as the reference, or near-straight RL/RRT paths would appear "better than optimal" (found on Day 1: raw RRT 2.57 rad vs raw A* 2.69 rad on the same query).
  - q₁ wraps around at ±π, so the grid is topologically a cylinder; the A* neighbors must handle this.
- **B4 — Artificial potential field** (P2–P3): a classical reactive baseline. It is included to show its local-minimum failures next to RL.
- **B5 — Computed-torque control** (P5): `τ = M̂(q)(q̈_ref + K_d ė + K_p e) + Ĉ(q,q̇)q̇ + ĝ(q) + B̂q̇`, tracking the B1 reference (P5a) or B2 reference (P5b). Gains K_p = 100, K_d = 20 **(default)** (critically damped). Evaluated with the exact model (M̂ = M) and with a perturbed model (masses ±20%).
- **B5' — PD + gravity compensation** (P5): `τ = K_p e + K_d ė + ĝ(q)`. A simpler model-based baseline that sits between computed-torque control and pure RL.
- **B6 — Replanning B2** (P4, Phase 2): replans whenever the current path becomes invalid.

### 7.3 C-space Tooling (unique to 2-DOF)

- `cspace.grid(scene)` returns a boolean occupancy grid, cached per scene.
- `cspace.components(grid)` returns connected components; used for scene validity.
- `cspace.distance_field(grid)` returns clearance in joint space; used for analysis and the optional distance reward.
- These grids are also used to plot obstacle regions, trajectories, value functions and policy vector fields.

### 7.4 Residual RL

Same formulation as UR5e spec §7.3 (`arm2dof/envs/residual.py`, `arm2dof/planners/base.py`; built Day 3 on the shared RRT-Connect, quintic and `WarmupSAC`):

- degraded base: circles inflated +5 cm except within 0.3 rad of start/goal, nearest IK branch only, a budget in collision checks, 20 shortcuts;
- **base budgets (decision 2026-10-27), calibrated so the base plans as often as the UR5e base does at 1500 / 3000 / 6000 checks on the S4 test set (77.5 / 84.5 / 90.0%):** P2 **{4000, 6400, 9000}**, P3 **{700, 2300, 3500}** checks (interpolated from 300 + 150 sampled episodes per stage; P2 plans 75 / 84 / 100% at 3200 / 6400 / 12800, P3 79 / 82 / 87–89% at 800 / 1600 / 3200 and plateaus near 95%, where inflation closes passages). Checks are far cheaper in the 2-D C-space, so reusing the UR5e numbers would not mean the same degradation. As on the UR5e, training draws the budget per episode from the stage's three; evaluation fixes one at a time; **fallback when it cannot plan: straight joint-space line to the IK goal that ignores collisions**, as on the UR5e;
- the tracker is wrap-aware in q₁; observation adds q_ref(t), q_ref(t+5) as (sin q₁, cos q₁, q₂/π);
- tested: references end at the goal, the fallback is a straight line, zero residual reproduces the base exactly.
- **Warm-up and α ramp scale with the run (decision 2026-10-29):** the planar runs use the UR5e §7.3 residual settings (w_res 0.1, α → 0.3, 50k-gradient-step critic warm-up, 150k-step ramp), which were sized for 1M-step E5 runs. At 0.25 gradient steps per transition the warm-up keeps the actor frozen until ~202k env steps, longer than a 150k-step PE3/PE4 run, so the PE3/PE4 residual would never have trained. PE3/PE4 therefore keep E5's **fractions** of the run: ramp 15% (22.5k steps), actor trains from 20% (critic warm-up 7k gradient steps, ending at 2k + 28k = 30k steps; `pe3.yaml`, `pe4.yaml`). Neither had started, so nothing is invalidated. PE8 (300k steps, two runs already finished) keeps the unscaled settings: its actor trains from ~202k, i.e. for the last third of the run, which is stated as a limitation of PE8.

Original Day-2 plan:

- `a = clip(a_base + α·π_res(s), −1, 1)`, α ramped 0 → 0.3 over the first 50k env steps (superseded: the UR5e §7.3 settings, scaled as above).
- **Dense reward, no HER.** The reference `q_ref` is planned for the original goal, so relabeled goals would be inconsistent with it (UR5e spec §7.3). The config loader rejects residual + HER.
- **Near-zero start in mean and noise:** near-zero init of the mean head; log-std bias −3; `ent_coef="auto_0.01"`; target entropy −4 (2 × −dim A).
- **`WarmupSAC`:** skips actor and entropy-coefficient updates for the first 20k gradient steps (critic-only warm-up; now 50k for E5 and PE8, 7k for PE3/PE4). Built here, reused unchanged for the UR5e.
- The base controller tracks a B2 reference planned with **inflated obstacles (+0.05 m)** and a **tight planning budget**. This makes the base reliable but conservative and suboptimal, which gives the residual something to improve.
- **P5 residual variant:** `τ = τ_B5'(perturbed model) + α·τ_max·π_res(s)`. This tests whether RL can compensate for model error on top of a model-based controller.

### 7.5 Demonstrations (Phase 2)

- 10k B3 (smoothed A*) trajectories per stage are used for BC pre-training and replay seeding.
- Ablation: compare demos from B2 (RRT) against demos from B3 (A*) to measure how much demo quality matters.

## 8. Experiments Matrix

| ID | Stage | Compares | Answers | Week 1? |
|---|---|---|---|---|
| PE0 | P0 | SAC vs TD3 vs PPO on `Reacher-v5` | Pipeline sanity | Optional |
| PE1 | P1 | DDPG/TD3/SAC+HER, PPO-dense, B1. Run as `pe1_v2` (q₁ wrap, 80 steps, goal curriculum, γ_her 0.98 for every HER agent; PPO-dense keeps the dense γ 0.99); v1 archived as `pe1_aborted_nowrap` (§6.2) | P-RQ1 | Yes |

**PE1 v2 interim (2026-10-27 04:50, 14 of 40 runs, full-distribution validation success at 1 cm, 50k steps):** SAC+HER 0.79 (n=3), TD3+HER 0.77 (n=4), DDPG+HER 0.44 (n=4, one seed at 0.05), **PPO-dense 0.00 (n=3; final distance 61.5 cm, i.e. no learning)**. PPO is reported as is, untuned (decision 2026-10-27): at the equal 50k-step budget PPO collects only ~12 rollouts (n_steps 128 × 32 envs = 4096 steps each), so the result says PPO is far less sample-efficient than the off-policy HER agents on P1, not that it cannot solve P1. The thesis states that the equal-step budget is what is being compared.
| PE2 | P1 | Sparse+HER vs dense; curriculum on/off, with the PE1-winning HER agent (`pe2.yaml`, rewritten by `pipeline_2dof.py`). The sparse+HER / curriculum-on cell is PE1's run, not rerun | Design choices → UR5e | Yes |
| PE3 | P2 | Pure RL vs B2 vs B4 vs residual. Run as SAC+HER (sparse) vs residual SAC (dense), mirroring E5 (`pe3.yaml`) | P-RQ3 | Yes |
| PE4 | P3 | Pure RL vs residual vs B2 vs B3. Same agents as PE3 (`pe4.yaml`) | P-RQ2, P-RQ3 | Yes |
| PE5 | P3 | Obstacle features: slots / distances / both | Design choice → UR5e (replaces UR5e E6 in Week 1) | Yes |
| PE6 | P3 | C-space value/policy maps for best agents | P-RQ4 | Yes |
| PE4b | P3 | **Recorded 2026-10-29 before training (user decision): the UR5e E5b redesign on the planar arm.** Residual SAC with an integrated joint offset, `o ← clip(o + α·π_res·Δq_max, ±0.3 rad)`; the base tracks `q_ref + s·o`, with s fading the offset to 0 within 5 cm of the goal (`envs/residual.py`, `mode="offset"`). Sparse reward (no HER), absorbing collision penalty, w_res, and warm-up / α ramp scaled to the run as in PE3/PE4 (7k gradient steps / 22.5k env steps). 5 seeds × 150k on the laptop (`pe4b.yaml`), queued after the current PE runs. Same evaluation as PE4 (p3_test, budgets 700 / 2300 / 3500, primary and final checkpoints), with its own rows in `results/tables/pe4.md`. In PE9-lite it is the "does the redesign transfer" comparison: offset vs additive hybrid, and offset hybrid vs base-only, on both arms (PE4b vs PE4, E5b vs E5). Tests (`tests/test_residual_offset.py`): offset persists and changes by ≤ α·Δq_max per step; zero effect at the goal; π_res = 0 equals base-only exactly. Speed: 87 steps/s on the shared laptop (~29 min per run). **Uses E5b v2 (2026-10-29, before PE4b completed):** fade = min(5 cm fade, linear fade over the last 10% of the reference), w_res on ‖o/0.3‖² (`pe4b.yaml`, run id `residual_offset_d60b3045`). The v1 runs (2 of 5 finished, 2 partial) are archived as `results/runs/archive/pe4b_v1_dist_fade/` and not reported. Tests: the fade reaches 0 by the end of the reference even when the tool is far from the goal; the penalty scales with ‖o‖ | P-RQ3, P-RQ5 | Yes |
| PE8 | P5a | SAC torque policy vs B5 vs B5' vs residual-on-B5' (exact / perturbed model). Configured (`pe8.yaml`): SAC+HER and residual-on-B5' (`envs/residual_torque.py`; dense, no HER, the E5 residual settings), 5 seeds × 300k steps (72.5 steps/s → ~69 min per run), queued after PE4. Both train on the nominal plant (as B5/B5' are designed on it); all four methods are evaluated on the exact and ±20% plants. **Interim (2026-10-28):** on the 100 training-validation episodes (nominal plant), B5' alone reaches 100% (0.45 cm; `2-dof/scripts/base_reference.py`), residual-on-B5' s0 57% at 300k (α = 0.3), SAC+HER s0 0%: on the nominal plant there is nothing for the residual to fix, and it degrades the base, as in UR5e E5 (§7.3 α sweep). The residual's case rests on the ±20% plants | P-RQ6 | Yes |
| PE9 | — | Rank agreement of design choices: PE2/PE5 vs UR5e E2. **PE9-lite (defined 2026-10-29, `scripts/pe9_lite.py`):** pairwise rankings on both arms with the same metric. SAC vs TD3 (PE1 vs E1); sparse+HER vs dense with the curriculum on (PE1/PE2 vs E2); curriculum on vs off (PE1/PE2 vs the 1-seed S1 probe, weak); hybrid vs base-only at the middle budget, and hybrid vs pure SAC+HER (PE3/PE4 vs E5). "Agree" = same option first; "clear" = 95% bootstrap CI over seeds of the difference excludes 0 with ≥ 3 seeds per trained side; a planner row (B2, B3, base@N) is one deterministic evaluation and counts as a fixed value (v0.12, 2026-10-30: read literally, the rule could never call a hybrid-vs-planner difference clear). PE5 did not run. | P-RQ5 | Yes (lite) |
| PE7 | P4 | Residual vs B6 | P-RQ3 (dynamic) | Phase 2 |
| PE10 | P5b | Torque policy with obstacles vs B5 on B2 reference | P-RQ6 | Stretch |
| PE11 | P3 | Residual + demos (B2 vs B3 demos) | Demo quality | Phase 2 |

## 9. Evaluation Protocol

### 9.1 Metrics

All UR5e metrics (UR5e spec §9.1) apply, except the orientation error. Planar-specific additions:

| Metric | Definition |
|---|---|
| Gap to best-known path (`arm2dof/eval/metrics.py`) | `L_joint(policy) / L_best − 1` on successful episodes, where per scene `L_best = min(L(A* + shortcut), L(RRT-Connect + shortcut))` (§7.2). Neither is a certified optimum after smoothing, so the metric is named for what it measures; it can be slightly negative if a policy beats both. |
| C-space clearance | Minimum joint-space distance to C-obstacle along the trajectory |
| Branch consistency | Fraction of episodes ending in the same elbow branch as the nearest-IK solution |
| Value–clearance correlation (P-RQ4) | Spearman ρ between learned V(q) and the C-space distance field |
| Tracking RMSE (P5) | Joint-space RMSE against the reference trajectory (for tracking controllers) |
| Peak torque, energy (P5) | `max |τ|`,`Σ τ²·dt` |
| Settling time (P5) | Time until the success condition (position + velocity) holds |
| Robustness gap (P5) | Success(exact model) − Success(perturbed model) |

### 9.2 Test Sets & Statistics

- Frozen `.npz` test sets: 1000 episodes per kinematic stage (cheap), 300 for P5a, plus a 300-episode out-of-distribution P3 set with more and larger obstacles.
- P5 perturbed-model test: the same 300 episodes with link masses scaled by ±20% (two copies).
- **P5a test set (built 2026-10-29):** episode i is the one `TorqueArmEnv` draws at reset seed 5,400,000 + i (i < 300), frozen to `testsets/p5a_test.npz` (disjoint from the P1–P3 test seeds 5.0M–5.3M and from training validation, 1,000,000 + i). **One success rule for all PE8 methods:** end effector within 1 cm and |q̇| < 0.1 rad/s for 5 consecutive steps within the 307-step horizon (the dense env's terminal condition). Every controller and the residual's base keep the nominal model on every plant. Also recorded: settling time, energy Σ τ²·dt, peak |τ|, and robustness gap = success(exact) − worst perturbed success (`scripts/eval_pe8.py` → `results/tables/pe8.md`).
- **Checkpoints and intervals (2026-10-29):** residual runs are evaluated at the UR5e §9.2 primary and final checkpoints (PE3/PE4: primary from 100k, since the actor trains from 30k; PE8: both are the 300k model), non-residual policies at best validation. Tables give the mean over seeds with a 95% bootstrap CI over seeds, plus a pooled Wilson interval labelled as episode-level only (UR5e §9.3; `scripts/pe_tables.py`).
- **Baseline check on 50 episodes (2026-10-29):** B5 and B5' reach 100% on the exact plant and drop to 10–30% on the ±20% plants. The perturbed plants leave both controllers just outside the 1 cm / 0.1 rad/s condition (final error ~2 cm), so the residual has a real error to remove there.
- A separate validation set is used for curriculum and model selection.
- Report mean ± 95% bootstrap confidence intervals over seeds, plus `rliable` IQM and performance profiles.

## 10. Visualization

**Interactive viewer** (`2-dof/scripts/play.py` + `arm2dof/viz/two_panel.py`; built 2026-10-27 in matplotlib only, so the same code exports GIF/MP4 offline. Left-click moves the goal, right-click adds/removes an obstacle; keys R/N/P/B/A/V as below. C-space marks the start, every IK branch of the goal, and A* to the branch the base plans to)

- **Left panel — workspace:** arm, obstacles with safety margins, goal, EE trail and the base reference path (residual mode).
- **Right panel — C-space:** occupancy grid, current q, the trajectory so far, and the A* optimal path for comparison.
- **P5 extras:** live torque bars per joint and a phase plot (q, q̇).
- **Interaction:**
  - drag the goal;
  - click to add or remove obstacles; drag obstacles;
  - `R` reset, `N` new scene, `P` pause, `B` toggle base-only / residual, `A` show A* path, `V` overlay the value-function heatmap on C-space, `M` toggle exact / perturbed model (P5).

**Colab / offline:** matplotlib animation to MP4/GIF (no EGL required), with the same two panels. Batch grid videos are made per method.

**Chapter figures:**

- C-space trajectories (RL vs RRT vs A*);
- value heatmaps and policy quiver fields;
- gap-to-best-known-path distributions;
- torque profiles and tracking error: RL vs computed-torque, exact vs perturbed model;
- learning curves.

## 11. Compute Plan

- CPU only. No GPU needed at any stage.
- Expected runtimes:
  - P1 runs: minutes each. *Measured (Day 3, shared 3×256 / 0.25 settings, 1 torch thread): HER agents ~66 steps/s (gradient-bound), PPO ~700 steps/s. PE1 therefore uses **50k env steps per run** for every agent (equal budget; P1 is FetchReach-scale): ~13 min per HER run, 40 runs ≈ 3 h on 2 workers.*
  - P3 runs: under about an hour each. *PE3/PE4: 150k env steps per run, 5 seeds (10 if time remains), 2 agents. Measured on P3 (12k-step smoke, wall clock incl. validation, under full load): residual SAC 60 steps/s, SAC+HER 76 steps/s, ~0.4 GB each → 33–42 min per run; 20 runs ≈ 12 h on the 1 worker the lane gets while E5 runs (2 otherwise). They queue behind PE1 and PE2 (`2-dof/scripts/pipeline_2dof.py`); the lane runs at most 1 worker while any E5 job runs and starts a job only when 2.5 GB RAM is free, so E5 keeps priority (UR5e spec §11).*
  - P5a runs (MuJoCo, 125 Hz): roughly 1–2 hours each.
- The whole Week-1 2-DOF matrix should fit in one or two overnight runs on the laptop, leaving Colab/Kaggle sessions free for the UR5e jobs.
- C-space grids are cached per scene in the test sets, so A* baselines are computed once.

## 12. Folder Structure

```
arm2dof/
├── SPEC.md
├── README.md
├── configs/
│   ├── env/   p1.yaml p2.yaml p3.yaml p4.yaml p5a.yaml p5b.yaml
│   ├── agent/ ddpg.yaml td3.yaml sac.yaml ppo.yaml residual.yaml
│   └── exp/   pe0.yaml … pe11.yaml
├── assets/
│   └── arm2dof.xml            # MuJoCo model: torque (P5) and position (P1 cross-check) variants
├── src/arm2dof/
│   ├── envs/      kinematic_env.py torque_env.py sampling.py geometry.py
│   ├── dynamics/  model.py                     # analytic M, C, g, Jacobian
│   ├── cspace/    grid.py components.py distance_field.py astar.py
│   ├── planners/  ik.py quintic.py rrt_connect.py smoothing.py potential_field.py
│   ├── control/   computed_torque.py pd_gravity.py
│   ├── agents/    factory.py residual.py demos.py bc.py
│   ├── eval/      metrics.py evaluate.py testsets.py      # stats reused from UR5e package where possible
│   └── viz/       viewer.py animate.py cspace_plots.py torque_plots.py
├── scripts/       train.py evaluate.py play.py make_testsets.py gen_demos.py
├── notebooks/     00_reacher_sanity.ipynb 01_train.ipynb 02_cspace_analysis.ipynb 03_dynamics.ipynb
└── tests/         test_fk_ik.py test_geometry.py test_cspace.py test_astar.py
                   test_dynamics.py test_reward_her.py test_env_api.py test_kinematic_vs_mujoco.py
```

**Stack:** same as the UR5e project, plus `pygame` and `matplotlib`. `scipy.ndimage` is used for C-space components and the distance field.

## 13. Tests (must pass before training)

- **FK/IK:** both IK branches satisfy FK within 1e-9; unreachable goals are detected.
- **Geometry:** segment–circle distance matches brute-force sampling within 1e-6.
- **C-space:**
  - every grid cell's occupancy agrees with direct collision checks;
  - q₁ wrap-around connectivity is correct.
- **A\*:** cost equals the Dijkstra optimum on the same 8-connected, q₁-wrapped graph; the path is collision-free; a route across q₁ = ±π is found when it is the only one; the smoothed A* reference (§7.2) is ≤ the raw RRT-Connect path length (+1 cell) on every test scene. (v0.3's "raw A* ≤ RRT" is false in general: the grid metric overestimates length.)
- **RRT-Connect (shared with the UR5e, `ur5rl/planners/rrt_connect.py`):** endpoints exact; collision-free at 10× finer re-check; wraps across q₁ = ±π; shortcut smoothing never lengthens the path.
- **Dynamics:**
  - analytic `M(q)`, `C(q,q̇)q̇`, `g(q)` match MuJoCo (`mj_fullM`, `qfrc_bias`) within 1e-6 at random states;
  - M is symmetric positive definite; `Ṁ − 2C` is skew-symmetric;
  - with zero torque and zero damping, total energy is conserved to integrator tolerance over 10 s;
  - **gravity acts on the P5 model:** `model.ngravcomp == 0`, no joint has `actuatorgravcomp`, and from rest with zero torque at q₁ = 0 the arm accelerates downward with `q̈ = −M⁻¹g(q)` (within 1e-6).
- **Computed-torque:** with the exact model and no torque saturation, it tracks a quintic reference with RMSE < 1e-3 rad.
- **HER reward:** batch equals per-sample; relabeling preserves collision, margin and energy penalties.
- **Residual mode:** same tests as UR5e spec §13 (config rejects residual + HER; zero residual reproduces `a_base`; mean |residual| at initialization < 0.02).
- **Cross-check:** the same P1 policy gives success within ±2% on the kinematic and MuJoCo environments.
- **API:** `gymnasium.utils.env_checker` and SB3 `check_env` pass.

## 14. One-Week Plan

The week is shared with the UR5e work; see UR5e spec §15 for the combined day-by-day table. 2-DOF tasks:

| Day | Deliverable | Background training |
|---|---|---|
| 1 | Kinematic env, FK/IK, geometry, C-space grid + A*, tests | PE1, PE2 (minutes each) |
| 2 | P2/P3 envs, RRT-Connect, potential field, residual wrapper, two-panel viewer | PE3, PE4, PE5 |
| 3 | Dynamics model + tests, P5a torque env, computed-torque and PD+g baselines | PE8 |
| 4 | Evaluate PE1–PE5, PE6 C-space figures → **design recommendations handed to UR5e S4** | — |
| 5 | PE9-lite once UR5e E2 results exist | (P5b stretch, if time) |
| 6–7 | Final evaluation, chapter figures and videos | — |

**Cut line, if behind schedule, in this order:** drop PE0 → drop B4 (potential field) → PE5 down to 2 variants → PE8 perturbed-model evaluation down to −20% mass only. **PE4 and PE8 core comparisons are never cut**: they carry the gap-to-best-known-path and dynamics results of this chapter.

## 15. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Kinematic env hides bugs that MuJoCo would expose | P1 kinematic-vs-MuJoCo cross-check test |
| Sign/index errors in the dynamics derivation | Analytic vs MuJoCo comparison, energy conservation, skew-symmetry tests |
| Torque policy unstable or chattering | 125 Hz control, torque-rate (smoothness) penalty, velocity term in success condition |
| Results on 2-DOF don't transfer to UR5e | Treat transfer as a measured question (PE9), not an assumption |
| Too easy to be interesting | Gap to best-known path, narrow-passage clutter, dynamics under model error |
| q₁ wrap-around breaks A* / metrics | Cylinder topology in grid; angle-wrapped joint distances everywhere |
| Time taken away from UR5e in a 1-week build | 2-DOF training runs on the laptop overnight; UR5e jobs get the cloud sessions; cut line above |

## 16. Resolved Decisions

| # | Question | Decision |
|---|---|---|
| 1 | Include dynamics (torque control)? | **Yes.** P5a and PE8 are in Week-1 scope; P5b (torque + obstacles) is a stretch goal. |
| 2 | Thesis placement | **Its own chapter**, framed as **methodology validation** for the UR5e work (outline in §1). |
| 3 | Timeline | 1 week, shared with the UR5e build (§14; UR5e spec §15). |

## 17. References

Same as the UR5e spec §18. Additions:

- Gymnasium `Reacher-v5` documentation (2-DOF MuJoCo reacher), used for P0.
- Khatib, *Real-Time Obstacle Avoidance for Manipulators and Mobile Robots*, IJRR 1986 (potential fields, B4).
- Kuffner & LaValle, *RRT-Connect: An Efficient Approach to Single-Query Path Planning*, ICRA 2000 (B2).
- Hart, Nilsson & Raphael, *A Formal Basis for the Heuristic Determination of Minimum Cost Paths*, 1968 (A*, B3).
- Spong, Hutchinson & Vidyasagar, *Robot Modeling and Control* (2-link dynamics, computed-torque control, PD + gravity compensation; B5, B5').
