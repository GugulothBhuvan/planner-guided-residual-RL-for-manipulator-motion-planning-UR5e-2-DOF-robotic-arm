# SPEC — RL Trajectory & Motion Planning for a 6-DOF UR5e Arm

| | |
|---|---|
| **Project** | MTP — Trajectory Planning of a 6-DOF Robotic Arm using Reinforcement Learning |
| **Author** | Bhuvan Raj, IIT Kharagpur |
| **Status** | Draft v0.24 — 2026-10-30 (v0.4: Day-1 measurements — integrated joint targets + gravity compensation, ±π limits, horizon filter, 3×256 at 0.25 grad/transition, per-machine throughput gate, lean HER buffer. v0.5: gravity compensation as a §6.1 modeling assumption; data-driven S2/S4 horizon rules in §4. v0.6: goal-independent observation + `GoalErrorExtractor`; S1/S2 horizon 150; explicit 3.2 rad goal bound + goal-distance curriculum; γ probe after the γ=0.95 E1 abort. v0.7: Day 3 — B1/B2 built, test sets frozen, S4 horizon 407, degraded base specified (endpoint-exempt inflation, check-count budget), residual stack built. v0.8: E5 reframed (cheap planner + residual matches expensive planner; primary figure = success/collision vs planning budget), base fallback = straight line ignoring collisions, S2 horizon 175, B1 velocity-capped quintic, disjoint per-stage validation seeds. v0.9: the E5 budget figure is computed on the 200-episode S4 test set only; small validation previews are sanity checks and are not reported. v0.10: update ratio fixed at 0.25 for every experiment, E5 included (the Colab 1.0 probe becomes a sensitivity check reported as a limitation); residual training draws the base budget per episode from {1500, 3000, 6000} checks; E5 runs on the laptop right after E1, E2 moves to Colab/Kaggle or runs after E5. v0.11: E5 starts in the slots E1's first wave frees, without waiting for E1; E1's second wave moves to Colab/Kaggle (or after E5); the 2-DOF lane runs 1 worker while E5 runs; PE3/PE4 at 5 seeds. v0.12: residual decision rule (§7.3), recorded before the 300k result was looked at; cause of the 04:51 stop (§11). v0.13: absorbing collision penalty (§6.7); E5/PE3 rerun with it and the §7.3 residual settings, v1 archived; checkpoints kept every 100k (§11). v0.14: E5 rerun launched despite the smoke FAIL (user option A); E2 seed 1 paused for E5, E2 reported as a shortfall on seed 0; P5a horizon 307 and P1 dual baseline reporting accepted. v0.15: conditional sparse-residual E5 arm (§7.3), recorded before the 200k/500k results). v0.16: E2 seed 1 (run on Colab) merged; E2 shortfall reported on both seeds; S2 contact rate without a contact penalty noted; second all-jobs stop and the DETACHED_PROCESS fix (§11)). v0.17: E5 seeds 1/2 and the sparse-residual arm move to Colab, seed 0 runs alone on the laptop (RAM gate 1.4 GB from measured peaks), 2-DOF lane paused while E5 runs, sparse rule decided at residual s0's 500k validation (§7.3, §11); α-sweep diagnostic (§7.3). v0.18: evaluation checkpoint rule for residual arms (§9.2), recorded before any test-set evaluation; laptop order residual s0 → SAC+HER s0 → sparse s0 → PE8, sparse s1/s2 on Colab/Kaggle; Windows power-throttling fix (§11). v0.19: E5 complete except sparse s0; evaluation before the rest of sparse s0; headline table with seed-level and pooled intervals (§9.3); incidents 3–4, job objects and the pipeline watchdog; launcher skips finished runs; PE3/PE4/PE8 seeds 2–4 on Kaggle (§11). v0.20: E5 reported as is; E5 diagnostic confirms the residual-cancellation mechanism; E5b (integrated offset residual) pre-registered (§7.3, §8). v0.21: E5b offset step α·π_res·Δq_max, v0 (rad units) archived and E5b restarted (§7.3). v0.22: E5b v1 diagnostic (saturated offset, fade never engages), v1 archived, E5b v2 recorded (§7.3). v0.23: E5b v2 is final; disk cleanup; E5b v2 test-set result (§7.3). v0.24: UR5e experiments frozen (§7.3)) |
| **Repo (proposed)** | `ur5e-rl-planner` |
| **Companion spec** | `arm2dof/SPEC.md` (planar 2-DOF methodology validation) |

> Numeric values marked **(default)** are starting points to be tuned and recorded in configs, not fixed decisions.

---

## 1. Problem Statement

Generate joint-space trajectories that drive a UR5e with a mounted gripper from its current configuration to a **target 6D tool pose (position + orientation)**. Trajectories must be **smooth**, **within joint limits**, and, from Stage 3 onward, **collision-free around obstacles**, with the gripper included in collision checking. A learned policy produces them closed-loop at a fixed control rate. The policy is compared against classical planners and against hybrid (residual) variants.

## 2. Goals / Non-Goals

**Goals**
- G1. A goal-conditioned UR5e + gripper simulation environment (MuJoCo, Gymnasium API) covering pose reaching and obstacle-aware motion planning.
- G2. A benchmark of model-free RL algorithms (TD3, SAC with HER; DDPG and PPO if time permits) on 6D pose reaching.
- G3. **Headline method (novel contribution, §3.1):** planner-guided residual RL for clearance-aware 6D pose motion planning.
- G4. A reproducible evaluation protocol with frozen test sets, multiple seeds and confidence intervals.
- G5. Visualization: an interactive local 3D viewer and rendered videos from Colab.
- G6. A simulation designed so that the real-robot stage (S6) can follow without rework: real-robot control interface, domain-randomization hooks, gripper geometry.

**Non-Goals (this phase)**
- Grasping or gripper actuation. The gripper is mounted, held at a fixed opening, and treated as rigid geometry for collision checking and the TCP frame.
- Perception from raw depth or point clouds. Obstacles are given as geometric primitives.
- Real-robot execution. Deferred to Phase 2 (S6, §4), after the simulation results are complete.
- Torque-level control on the UR5e. The policy commands joint position increments. Dynamics are studied on the 2-DOF arm (companion spec).

## 3. Research Questions

| ID | Question | Primary metric | Phase |
|---|---|---|---|
| RQ1 | Which model-free algorithm (TD3/SAC+HER; DDPG/PPO if time) is most sample-efficient and accurate for 6D pose reaching on UR5e? | Success rate vs env steps; final pose error | Week 1 |
| RQ2 | How does reward type (sparse+HER vs dense) affect pose accuracy? Orientation representation (quat vs 6D) as stretch. | Final orientation error; success at tight tolerance | Week 1 |
| RQ3 | Can a learned residual let a cheap planner match an expensive one in cluttered scenes? | Success and collision rate vs planning budget (primary); jerk, clearance, time (secondary) | Week 1 (headline) |
| RQ4 | Do planner demonstrations (BC pre-training) reduce the samples needed? | Env steps to reach 80% success | Phase 2 |
| RQ5 | How much performance survives transfer to the real UR5e? | Success / error / collisions: sim vs real | Phase 2 (S6) |

**Hypotheses**
- H1. SAC+HER ≥ TD3+HER on success and on variance across seeds.
- H2. Dense reward with curriculum reaches tight 6D tolerances faster than sparse+HER alone.
- H3. **The residual lets a cheap planner match an expensive one:** at a small planning budget (1500–6000 collision checks), base + residual reaches the success and collision rate of full B2 (2 s budget, exact obstacles, 200 shortcuts), while the base alone at the same budget does not. Pure RL is the lower reference. Jerk, clearance and time are secondary.

### 3.1 Novel Contribution (minimum novelty required by guide)

**Planner-guided residual SAC for clearance-aware 6D pose motion planning on a UR5e with gripper.** The claimed contribution has three parts:

1. **A learned residual lets a cheap planner match an expensive one.** The base controller tracks an RRT-Connect path planned with inflated obstacles and a small budget counted in collision checks; when that budget runs out, the base falls back to a straight joint-space line to the IK goal that ignores collisions. The RL residual learns to repair the cheap base (avoid what the fallback line hits, recover pose accuracy, use clearance well), so that base + residual matches full B2 in success and collision rate at a fraction of the planning cost. The **primary result** is success and collision rate vs planning budget (1500, 3000, 6000 checks) for base-only and base + residual, with full B2 as the reference; the residual is trained with the budget drawn per episode from {1500, 3000, 6000} and evaluated at each fixed budget. Jerk, clearance and time are secondary. Because the base already supplies successful trajectories, the residual agent trains on the dense reward **without HER** (§7.3 explains why HER is incompatible with a goal-specific reference).
2. **HER-compatible reward decomposition for the goal-conditioned (non-residual) agents.** Goal-independent safety terms (collision, clearance margin, smoothness) are stored at collection time and re-added during hindsight relabeling. Standard HER silently drops these terms. This applies to the pure-RL baselines in E2 and E5, which must solve clutter without a planner. The same principle covers observations: goal-dependent features must be recomputed from the relabeled goal (done in the policy by `GoalErrorExtractor`, §6.4); E1's first launch failed exactly this way.
3. **Validation methodology.** Every design choice is first validated on a planar 2-DOF arm, where exact optimal paths exist and the optimality gap can be measured (companion spec). The UR5e chapter then reports which choices transferred.

**Honest scoping:** residual RL (Silver et al.; Johannink et al.) and HER exist individually. The novelty is their combination with a degraded sampling-based planner for 6D pose-constrained motion planning, the reward decomposition, and the 2-DOF → 6-DOF validation protocol. The thesis must state this explicitly, including the design decision that HER and the residual reference are kept apart.

## 4. Task Ladder (Stages)

| Stage | Goal | Obstacles | Episode length (default) | Target | Phase |
|---|---|---|---|---|---|
| S0 | panda-gym Reach (tutorial replication) | — | 50 | Reproduce tutorial curve | Week 1, optional background run |
| S1 | Position only | — | **150 steps (6 s)** | ≥95% success at 2 cm | Week 1 |
| S2 | Position + orientation | — | **175 steps (7 s)** | ≥90% success at 1 cm / 5° | Week 1 |
| S3 | Pose | 1–3 static, fixed layout | 150 | Debug stepping stone only (no experiment) | Week 1 |
| S4 | Pose | 1–K static, randomized each episode | **407 steps (16.3 s)** = max(200, ⌈1.25 × p99 B2 duration⌉) (see below) | Base + residual at ≤ 6000 collision checks matches full B2's success and collision rate (within the 95% CI), where base-only does not | Week 1 (headline) |
| S5 | Pose | Moving obstacles (≤0.2 m/s) | 200 | Collision rate below replanning baseline | Phase 2 |
| S6 | Pose, real UR5e | Static, measured layout | — | Sim-to-real gap report (RQ5) | Phase 2 |

**Horizon rules (data-driven, recorded here when applied):**
- **S1/S2 (applied 2026-10-26): 150 steps (6 s).** The goal bound stays at 3.2 rad (§6.5), so 6 s is a 1.5× time margin over the 3.2 s a full-speed move needs; the Day-1 oracle's slowest S2 success had been 3.84 s of 4 s. The goal bound is now an explicit distance, no longer derived from the horizon. E1's failures are still classified after it finishes (`scripts/failure_breakdown.py`: *timeout* = distance still decreasing at the horizon, *stuck* = not decreasing and no contact, *collision* = contact in the final 25 steps), with a 1.5× horizon replay of the failed episodes.
- **S2 (applied 2026-10-27, before E2): 175 steps (7 s).** B1 now uses the shortest quintic under the arm's 0.98 rad/s cap, T = 1.875 · max|Δq| / v_max, which is ≤ 6.12 s for the 3.2 rad bound; 7 s leaves room for tracking lag. B1 on the 100 S2 validation episodes: 99% success, longest 6.12 s. The S2 test and validation sets carry the new horizon (episodes unchanged). S1 stays at 150 (E1 is running on it).
- **S4 (applied 2026-10-27): 407 steps.** B2 (RRT-Connect + shortcut smoothing + time-scaling) was run on all 200 S4 test episodes: success 99.5%, duration p50 6.16 s, p99 13.00 s, max 13.2 s → `max(200, ⌈1.25 × 13.00 / 0.04⌉)` = 407 steps. B2's planning time is excluded from the duration (it is the latency metric). The horizon is stored in the test-set metadata and in `configs/env/s4.yaml` (written by `scripts/make_testsets.py`).

**E1 history (S1).**
1. *Launch 1:* goal errors inside `observation` broke HER (§6.4). 0% validation success at 120k–160k steps; archived as `results/runs/e1_aborted_goal_in_obs`.
2. *Launch 2:* γ = 0.95 with HER. Its effective horizon 1/(1−γ) ≈ 20 steps is far below the ~80 steps a 3.2 rad goal needs, so the sparse-reward value signal is flat for distant goals. Validation success 0–6% at 200k–250k steps (final distance 35–46 cm). The same pipeline on near goals (≤ 0.6 rad, 5 cm tolerance) reached 93% at 40k, which ruled out a pipeline bug. Archived as `results/runs/e1_aborted_gamma095`.
3. *Probe (SAC+HER, seed 0, 120k steps, full-distribution validation):* (a) γ 0.98 + goal-distance curriculum, (b) γ 0.99 + curriculum, (c) γ 0.98 alone. Winner = highest success at 120k; within 0.03 counts as a tie, and ties go to the simpler config (c, then a, then b). If every probe is below 0.20, E1 and E2 use 500k steps per run and the shortfall is reported. E1 is relaunched with the winner's settings for both algorithms, then E2 is queued (`scripts/pipeline_e1_e2.py`; decisions in `results/runs/pipeline_decisions.json`).

**1-week rule:** stages are **time-boxed, not gated**. If a stage misses its target by its day in §15, proceed with the best result and report the shortfall honestly.

## 5. System Architecture

```
               ┌──────────────────────────── training (Colab / Kaggle CPU) ────────┐
 configs/*.yaml│                                                                   │
      │        │  TestSets ──► Evaluator ──► metrics.json / plots                  │
      ▼        │     ▲                                                             │
 ┌──────────┐  │  ┌──┴───────────┐   obs/goal   ┌───────────────┐                  │
 │ Scene XML│──┼─►│ UR5ePoseEnv  │─────────────►│ Agent         │                  │
 │ UR5e +   │  │  │ (MuJoCo,     │◄─────────────│ SAC/TD3 (+HER)│                  │
 │ 2F-85 +  │  │  │  Gymnasium   │    action    │  ± residual   │◄── a_base ──┐    │
 │ table +  │  │  │  GoalEnv)    │              └───────────────┘             │    │
 │ obstacles│  │  └──────────────┘                                   ┌────────┴──┐ │
 └──────────┘  │         ▲                                           │ Classical │ │
               │         └──── collision checks / FK ────────────────│ IK, quintic│ │
               │                                                     │ RRT-Connect│ │
               │                                                     └───────────┘ │
               └──────────── checkpoints → Google Drive ───────────────────────────┘
                                         │
                ┌────────────────────────┼─────────────────────────┐
                ▼                        ▼                         ▼
      Local viewer (laptop)    Colab offscreen renderer    Phase 2: URSim → real UR5e
      mujoco.viewer            MP4 via EGL + mediapy       (ur_rtde joint streaming)
```

## 6. Environment Specification

### 6.1 Simulation
- **Engine:** MuJoCo ≥ 3.x (Python bindings).
- **Robot:** UR5e from `mujoco_menagerie/universal_robots_ur5e`.
- **End-effector:** Robotiq 2F-85 from `mujoco_menagerie/robotiq_2f85`, attached to the tool flange (MjSpec attach).
  - **Rigid gripper (Week 1):** the gripper is welded to the flange as one rigid body at a fixed opening, with no joints, actuators or equality constraints. No grasping in this phase. The unmodified menagerie 2F-85 (linkage, equality constraints, actuator) is reserved for Phase 2 grasping.
  - **Gripper collision geometry:** exactly 2 capsules sized to enclose the 2F-85 visual mesh. The visual meshes are kept for rendering only.
  - **TCP:** a site at the midpoint between the fingertip pads. All goals, errors and metrics use this TCP frame. Its offset is read from the model, not hard-coded.
  - The 2 gripper capsules take part in obstacle, table and self-collision checks.
- **Scene:** robot mounted on a table at z = 0. Floor and table are collision geoms.
- **Physics timestep:** 0.002 s **(default)**. **Frame skip:** 20, giving a control period of 0.04 s (25 Hz).
- **Collision geometry:** primitive capsules per arm link and the 2 gripper capsules above, used for robot–obstacle checks (fast and MJX-compatible). Visual meshes are used only for rendering.
- **Modeling assumption — gravity compensation:** every arm body and the gripper have `gravcomp = 1`, applied through the actuators (`actuatorgravcomp`). This models the UR controller, which compensates gravity internally, so a streamed joint target is held without sag (Day 1: steady-state error 0.13 mrad with it, 15–32 mrad without, i.e. 2–3 cm at the TCP). Consequences:
  - joint dynamics seen by the policy are those of a gravity-compensated position servo, not of a bare arm; torque-level dynamics are studied only on the 2-DOF P5 model, which has **no** gravity compensation (companion spec §5.3);
  - set in `scripts/build_scene.py` (`GRAVITY_COMPENSATION`), scoped to the UR5e scene only;
  - S6 checks the assumption: hold error on the real arm vs sim.

### 6.2 Action Space
- `a ∈ [-1, 1]^6`, interpreted as normalized increments of the **joint target** (integrated targets).
- Each control step:
  1. `q_target ← clip(q_target + a · Δq_max, q_min, q_max)`, with `Δq_max = 0.04 rad/step` (= 1 rad/s at 25 Hz) **(default)**;
  2. `q_target ← q + clip(q_target − q, −0.25, 0.25)` (rad), so the target leads the arm by at most 0.25 rad and cannot wind up while the arm is blocked;
  3. `q_target` is sent to the model's position actuators.
- `q_target = q` at reset.
- **Why integrated targets:** the menagerie servos have kv/kp = 0.2 s. Re-anchoring the target on the measured q every step (v0.3 rule) capped the joint speed at Δq_max/0.2 s ≈ 0.2 rad/s (measured 0.18–0.25 rad/s), leaving ~0% of S1/S2 goals reachable within the horizon. With integrated targets the measured speed at |a| = 1 is 0.98 rad/s on every joint (Day-1 check, `scripts/check_reachability.py`).
- **Gravity compensation:** the arm and gripper bodies use MuJoCo actuator gravity compensation, as the UR controller does internally. Without it the servos settle up to ~30 mrad (2–3 cm at the TCP) below every target, which the real arm does not do.
- **Joint limits (Week 1): ±π on all joints** (config `joint_limit`), narrowed from the model's ±2π to remove wrap-around ambiguity.
- This interface (a stream of joint targets) is chosen so that S6 can send the same targets to the real arm, interpolated up to the controller rate.

### 6.3 Goal Representation
- `goal = [p (3), r6d (6)] ∈ R^9` for the TCP, where `r6d` is the first two columns of the rotation matrix (Zhou et al., 2019).
- Converting back: Gram–Schmidt on the two columns, then cross product for the third.
- A quaternion variant (`[p, q]` with q canonicalized so w ≥ 0, ∈ R^7) is kept behind a config flag for the RQ2 stretch ablation.

### 6.4 Observation (dict, Gymnasium GoalEnv)

| Key | Content | Dim |
|---|---|---|
| `observation` | sin q, cos q | 12 |
| | q / q_limit (normalized; shows proximity to joint limits) | 6 |
| | q̇ (scaled) | 6 |
| | TCP position, TCP r6d | 9 |
| | previous action | 6 |
| | target lead (q_target − q) / 0.25 (§6.2) | 6 |
| | *(S3+)* obstacle slots: K × [center 3, radius 1, mask 1], K=4 **(default)** | 20 |
| | *(S3+)* min distance to nearest obstacle for 6 links + gripper, clipped at 0.3 m (analytic, §6.5) | 7 |
| | *(residual only)* reference joint config q_ref(t) and q_ref(t+5) | 12 |
| `achieved_goal` | TCP [p, r6d] | 9 |
| `desired_goal` | target [p, r6d] | 9 |

All features are normalized to roughly unit scale. Running observation normalization (VecNormalize) is used for PPO only.

**Rule: goal-dependent observation features must be computed from the relabeled goal.** This is the same family as the safety-term decomposition (§6.7): HER changes the goal of a stored transition, so every quantity that depends on the goal must be recomputed from the new goal (rewards, goal-relative features), and every quantity that does not must be carried over unchanged (safety terms, the goal-independent `observation`). A feature stored with the original goal silently corrupts relabeled samples.

**`observation` must not depend on the goal.** HER relabels `desired_goal` only; any goal-derived feature inside `observation` would describe the original goal on relabeled samples, while the reward describes the new one. v0.4 listed position and rotation error here, and the first E1 launch was invalid because of it: 0% validation success at 120k–160k steps, stopped and archived as `results/runs/e1_aborted_goal_in_obs`. Goal errors are now computed **inside the policy** by `GoalErrorExtractor` (`agents/policies.py`): features = [observation, achieved_goal, desired_goal, desired − achieved] (position error 3 + r6d difference 6). They are recomputed for every relabeled goal. A test asserts that `observation` is unchanged when the goal changes (§13).

### 6.5 Goal & Scene Sampling
- **Goals are sampled in joint space, then mapped with FK:** draw a random q within limits, reject it if the arm or gripper self-collides, goes below the table or (S3+) collides with obstacles, and use FK(q) of the TCP as the goal pose. This guarantees every goal is reachable and collision-free.
- **Initial configuration:** a nominal "home" pose plus uniform noise (±0.3 rad) **(default)**.
- **Goal bound (S1/S2):** `max_j |q_goal,j − q_start,j| ≤ 3.2 rad` (config `goal_max_joint_dist`; = 0.8 × 1 rad/s × 4 s, the Day-1 horizon). Since the S1/S2 horizon became 6 s (§4) the bound is an explicit distance, not derived from the horizon. Goals are sampled **uniformly over that set directly**: it is a box (joint limits ∩ start ± bound), so no rejection against the full joint box is needed. S4 has no bound.
- **Goal-distance curriculum (S1/S2, config `curriculum`):** the bound starts at 0.6 rad and rises by 0.3 rad whenever the success rate of the last 100 training episodes **whose goal came from the current level** exceeds 80%, up to the 3.2 rad cap. 20% of training goals always use the full 3.2 rad bound; those episodes do not count toward the trigger (counting them caps the rate near 0.8·p_level + 0.2·p_full and stalls the curriculum; probe runs a/b were restarted from their 50k checkpoints with this fix). The level is logged (`curriculum/level`) and persisted for resume. **Validation and test episodes always use the full distribution.** Whether it is used in E1/E2 is decided by the S1 probe (§4).
- **Reachability check (Day 1, 500 episodes per stage):** an oracle driving full speed straight to the sampled goal configuration through the §6.2 action rule.

  | Stage | Filter | Oracle success at episode end | Among contact-free episodes | Oracle touched table/self | Latest success / horizon |
  |---|---|---|---|---|---|
  | S1 | 3.2 rad | 96.4% | **100%** | 11.4% | 3.68 s / 4 s |
  | S2 | 3.2 rad | 96.4% | **100%** | 11.4% | 3.84 s / 4 s |
  | S4 (timing only, obstacles off) | — | 82.6% | **100%** | 24.8% | 7.24 s / 8 s |

  The oracle's straight joint-space line sometimes touches the table or the arm itself. Those goals are still reachable by another route; the policy (or B2) has to find it.
- **Obstacles (S4):** **spheres only in Week 1** (boxes in Phase 2), sampled in the workspace annulus between the start and goal TCP positions. Reject a sample if it intersects the start or goal arm configuration. When the test set is built, RRT-Connect verifies that at least one collision-free path exists.
- **Clearance computation (cost control):** body–obstacle distances for the observation and margin reward are computed **analytically in NumPy**, not with per-pair `mj_geomDistance` calls. Each arm link is a capsule and the gripper's collision geometry is exactly 2 capsules (§6.1), so each step is 8 segment–point distances × K spheres, vectorized. Because these are the model's actual collision geoms, the analytic distances match `mj_geomDistance` exactly for links and gripper (§13). MuJoCo contacts are still used for the collision flag (they come free with `mj_step`).
- **Step-time targets (soft, profiled on Day 1):** a full env step ≤ 2 ms, profiled separately for physics (`mj_step` × frame skip) and clearance (≤ 0.1 ms). These are diagnostics, not gates. The hard gate is end-to-end SAC throughput (§11).

### 6.6 Errors & Success
- `d_pos = ‖p_goal − p_tcp‖₂`
- `θ = arccos((tr(R_goalᵀ R_tcp) − 1) / 2)` (geodesic angle, clipped for numerical safety)
- **Success:** `d_pos < ε_p` and `θ < ε_θ`.
- **Tolerance curriculum (S2+):** starts at ε_p = 5 cm, ε_θ = 30° and tightens linearly to 1 cm / 5° as the rolling success rate exceeds 80%. Evaluation always uses the final tolerances.
  - **Week 1: not used.** E2 runs with the goal-distance curriculum when the S1 probe selects it (§6.5), and with fixed final tolerances. Running two success-gated curricula at once would make them compete for the same rolling-success signal and blur which one helped. The tolerance curriculum is tested in E3 (Phase 2), as planned.

### 6.7 Rewards
Two reward modes, selected by config.

**Sparse (with HER):** `r = 0` if successful, else `−1`, plus the goal-independent safety terms below (S3+). The function is vectorized over batches of (achieved, desired) goals so HER can relabel.

**Dense:**
```
r = − w_p · d_pos − w_r · θ
    − w_a · ‖a‖² − w_s · ‖a_t − a_{t−1}‖²          (effort, smoothness)
    − w_m · Σ_bodies max(0, d_safe − d_body)        (safety margin, S3+; 6 links + gripper)
    − w_c · 1[collision]                            (S3+, also terminates)
    + R_succ · 1[success]
```
Defaults: `w_p=1.0 /m, w_r=0.3 /rad, w_a=0.01, w_s=0.05, w_m=5.0, d_safe=0.05 m, w_c=10, R_succ=5`.

**Absorbing collision penalty (decision 2026-10-28; replaces the one-off `−w_c` above).** Episodes still end on collision (S3+), but the collision step pays the value of staying in the worst state forever:

```
R_coll = r_min / (1 − γ) − w_c
```

with `r_min` the analytic per-step minimum of the reward mode in use and `γ` the agent's discount (`train.py` copies it into the env config on obstacle stages: γ_her = 0.98 for sparse + HER, γ = 0.99 for dense). Why: with a one-off −10, ending an episode by crashing was worth more than failing slowly, and pure SAC+HER learned exactly that (0% success, 100% collisions after ~9 steps; v1 runs archived as `e5_v1_crash_exploit`). A penalty depending on the remaining steps (T − t) was rejected: t is not in the observation, so that reward would be non-Markov.
- `r_min`, UR5e: sparse `−1 − w_s·24 − w_m·7·d_safe` = −3.95 → **R_coll = −207.5**; dense `−w_p·2R − w_r·π − w_a·6 − w_s·24 − w_m·7·d_safe` = −6.54 → **R_coll = −663.8**. Here 24 = max ‖a_t − a_{t−1}‖² and 6 = max ‖a‖² for a ∈ [−1,1]⁶, 7 bodies, and R = 1.293 m is the shoulder-to-TCP reach bound (arm lengths after the shoulder + TCP offset; the sampled maximum is 1.088 m). The margin term clips clearance at 0 (penetration is the collision term's job), so it is bounded by `d_safe` per body.
- The sparse value is lower than the −1/(1−γ) − w_c = −60 first proposed, because on obstacle stages the sparse per-step reward also carries the smoothness and margin terms (down to −2.95 together). With −60, a policy that times out while maximising those terms would still do worse than one that crashes, and "crashing is always worse than timing out" would not hold.
- Planar arm: the same rule (2 joints, 2 links, d_max = 1.8 m): sparse −105.0, dense −282.0.
- HER: the collision step's total reward is exactly R_coll for the episode's goal (the goal-independent part is stored as `R_coll − goal term`); on a relabeled goal it can differ by the goal term's range (sparse: 1), still far below any time-out.
- Residual agents (§7.3): their L2 penalty is charged as its worst case forever on a collision step (`−w_res·6/(1−γ)`), so the bound holds with it.
- Tests (§13): `r_min` bounds every observed per-step reward under bang-bang actions; the env pays exactly R_coll on a collision step; R_coll < r_min·(1−γᵏ)/(1−γ) for every k (crashing is worse than surviving any number of further steps at the worst reward), in both modes, on both arms.

**HER reward decomposition (contribution part 2):** collision, margin and smoothness penalties do not depend on the goal. They are stored in `info` at collection time, and `compute_reward(achieved, desired, info)` adds them back after relabeling. This must be unit-tested.

**Which agents use HER:** pure-RL agents (E1, E2, E5 pure SAC) only. Residual agents use the dense reward without HER (§7.3). The config loader rejects `residual: true` together with `her: true`.

### 6.8 Termination & Truncation
- **Truncation:** at the stage episode length.
- **Termination (dense mode):** on success held for 5 consecutive steps, or on collision (S3+).
- **Sparse/HER mode:** fixed horizon with no early success termination, which matches standard HER practice. Collisions still terminate in S3+.

### 6.9 Sim-to-Real Hooks (built now, off by default)
Config flags exist from the start so S6 needs no environment rewrite:
- action latency of 0–2 control steps;
- joint position noise σ = 0.002 rad;
- obstacle pose noise σ = 1 cm;
- control-period jitter ±10%;
- gripper mass and inertia ±20%.

Week 1 trains without randomization. One robustness evaluation with the hooks switched on is run in E8.

## 7. Methods

### 7.1 RL Agents (Stable-Baselines3)

| | TD3 | SAC | DDPG *(if time)* | PPO *(if time)* |
|---|---|---|---|---|
| Replay / HER | min(1e6, run steps), HER future k=4 (pure-RL only) | same | same | — |
| Network | [256,256,256] | same | same | [256,256] π / v |
| LR | 1e-3 | 3e-4 | 1e-3 | 3e-4 |
| Batch | 256 | 256 | 256 | minibatch 256 |
| Update ratio | **0.25 gradient steps per transition** (4 envs → `train_freq=1, gradient_steps=1`) | same | same | — |
| γ | HER: 0.98 or 0.99 from the S1 probe (§4; 0.95 was too short-sighted) / 0.99 (dense) | same | same | 0.99, λ=0.95 |
| τ | 0.05 | 0.05 | 0.05 | — |
| Exploration | σ=0.2, target noise 0.2 | auto-entropy | Gaussian σ=0.2 | — |
| Envs | 4 | 4 | 4 | 16–32 (SubprocVecEnv) |

These are starting points taken from common goal-conditioned settings. Every change is logged in the run config.

**Update ratio, network and batch size are a deliberate compute trade-off (§11).**
- Configs state the ratio **per transition**. SB3 counts `train_freq` in vec-env steps (n_envs transitions each), so the factory sets `gradient_steps = ratio × n_envs` with `train_freq=1`. (v0.3's `train_freq=4, gradient_steps=2` would have been 0.125 per transition with 4 envs.)
- Chosen on Day 1 by the rule "2×256 at 0.5 if the laptop's total over 4 parallel runs is ≥ 150 steps/s, else 3×256 at 0.25". Measured: 2×256 at 0.5 gave 122.5 steps/s total, so **3×256 at 0.25** is used (216 steps/s total over 4 runs).
- **Update ratio: 0.25 for every experiment, E5 included (decision 2026-10-27).** The Colab probe below is no longer a gate. It becomes a sensitivity check: its result is reported in the thesis as a limitation (how much faster a higher ratio learns on S1), not used to change any experiment.
- *Sensitivity check (formerly the E5 gate):* probe a's config (SAC+HER, γ_her 0.98, goal curriculum, 150-step horizon) rerun on Colab at **1.0** gradient steps per transition, 120k steps (`configs/exp/probe_s1_grad1.yaml`, `notebooks/colab_probe_grad1.ipynb`), compared with probe a at equal env steps by `scripts/compare_runs.py`. Rule, fixed before the result: *clearly faster* = success at 120k ≥ probe a + 0.10, or mean final position error over the last two evaluations ≤ 0.75 × probe a's. The rule is kept only to phrase the reported result ("clearly faster" or not).
- 0.25 gradient steps per transition may cost sample efficiency. It is applied **identically to every agent**, so comparisons stay fair, and the thesis states it as a limitation.
- **Replay buffer size** is capped at the run length: a larger buffer never fills, and on Windows SB3 buffers are fully resident and every checkpoint pickles the whole buffer.
- **HER buffer:** `SafetyHerReplayBuffer` (`agents/her.py`) stores only the goal-independent term `r_indep` (8 B per transition) and passes it to `compute_reward` on relabeling. SB3's default passes empty infos (the safety terms would be dropped; our `compute_reward` raises instead), and `copy_info_dict=True` costs ~340 B per transition plus a deep copy per sample.
- **Training envs load without visual meshes** (`discardvisual`): 1 MB instead of 98 MB per model, bit-identical physics.

### 7.2 Classical Baselines
- **IK:** analytic UR5e inverse kinematics for the TCP (up to 8 solutions), with the gripper offset composed onto the flange. Pick the solution nearest to the current q within limits. Numerical IK (e.g. `mink`) is the fallback.
- **B1 — IK + quintic:** joint-space quintic interpolation from q₀ to q_goal with zero boundary velocity and acceleration. Duration T comes from per-joint velocity and acceleration limits. Valid for S1–S2.
  - Duration = the shortest quintic under the arm's 0.98 rad/s cap, **T = 1.875 · max|Δq| / v_max** (velocity cap only). A quintic peaks at 1.875× its mean speed, so a 3.2 rad move takes 6.12 s. (Day-3 first version used v 0.9 / a 3.0 and needed up to 6.6 s.) B2 keeps v 0.9 / a 3.0 rad/s², with which the S4 horizon was derived.
- **B2 — RRT-Connect + shortcut smoothing + time parameterization:** a joint-space planner using MuJoCo collision checks (gripper included). Followed by 200 random shortcut iterations and quintic time-scaling per segment. Planning time is capped at 2 s and counts as a failure if exceeded.
  - Shared RRT-Connect (`planners/rrt_connect.py`, also used by the 2-DOF arm): extend step 0.3 rad, **edge collision check every 0.01 rad** (Day 3: at 0.05 rad, 1 of 18 tracked B2 executions collided because checked points were up to ~5 cm apart at the TCP); goal = nearest collision-free analytic-IK branch first, others tried within the budget.
- **Baselines execute through the env**, via the same action interface as the policy: `ReferenceTracker` applies `a = clip((q_ref(t+1) − q_target)/Δq_max, −1, 1)` (also the residual base controller).
- **B3 (S5, Phase 2):** B2 with replanning whenever the current path becomes invalid.

### 7.3 Residual RL (headline)
- **Base controller:** tracks the B2 reference trajectory.
  `a_base = clip((q_ref(t+1) − q_target) / Δq_max, −1, 1)`.
  Under integrated targets (§6.2) this moves the target onto the next reference point whenever the reference advances by at most Δq_max per step; the servo then tracks the target. (v0.3 used `q_ref(t) − q`, which assumed targets re-anchored on the measured q.)
- **Policy output:** `a = clip(a_base + α · π_res(s), −1, 1)`, with α ramped linearly from 0 to 0.3 **(default)** over the first 50k env steps.
- **Base agent:** SAC on the **dense reward, without HER**.
  - **Why no HER:** the observation contains `q_ref(t)` and `q_ref(t+5)`, and the base action tracks a path planned to the *original* goal. A relabeled goal would not match the reference or the base action in the same transition, so the critic would learn from inconsistent tuples. Recomputing the reference per relabeled goal would need an RRT call per sample, which is infeasible.
  - **Why it's unnecessary:** the base controller already produces successful and near-successful trajectories, so the sparse-reward exploration problem that HER solves does not arise.
  - **Phase 2 ablation (optional):** residual + HER restricted to relabel goals that lie on the reference path.
- **Starting at the base controller's behavior** requires both the mean and the noise of the residual to be near zero:
  - the last layer of the actor's **mean** head is initialized near zero;
  - the **log-std** head bias is initialized to −3 (σ ≈ 0.05), so sampled residuals are about ±0.015 at α = 0.3 instead of ±0.2–0.3;
  - the entropy coefficient starts at 0.01 (`ent_coef="auto_0.01"`), and the target entropy is set to −12 (2 × −dim A). With SB3's default of −6, auto-tuning would immediately push σ back up.
- **Critic warm-up:** a small `WarmupSAC` subclass (`agents/residual.py`) freezes the actor and entropy coefficient for the first 20k gradient steps **(default)**, so early actor updates don't destroy the base behavior. Implemented by setting their learning rates to 0 after SB3's schedule update (Adam then leaves the parameters exactly unchanged); tested.
- **Degraded base** (otherwise there is nothing for the residual to learn), as implemented on Day 3 (`envs/residual.py`, `planners/baselines.py`):
  - obstacles inflated by +5 cm **except within 0.3 rad (max-norm) of the start and goal configuration**, where the exact check applies. Episodes guarantee only 2 cm clearance at the endpoints, so full inflation made the start or goal itself "collide": 18 of 25 base failures on 60 S4 episodes were that, unrelated to path quality.
  - planning budget counted in **collision checks** (6000 ≈ 0.2 s on the Day-3 laptop), not wall-clock: a wall-clock budget made base reliability depend on machine load, which is irreproducible inside a training env. B2's own 2 s cap stays wall-clock (latency is its metric).
  - nearest IK branch only; 20 shortcut iterations.
  - Measured on 60 S4 episodes: the base plans 50 (83%); every planned one succeeds when executed base-only, with no collisions. Its trajectories are only slightly longer than full B2 (median 5.96 s vs 5.86 s), so at 6000 checks the residual's room is mainly the 17% of episodes with no base plan, plus clearance and smoothness; at 1500/3000 checks the base fails more often and the residual carries more.
  - **Training budget (decision 2026-10-27):** each training episode draws its base budget uniformly from {1500, 3000, 6000} checks (`max_checks: [1500, 3000, 6000]` in `configs/agent/residual.yaml`; drawn from the env's seeded RNG, logged as `info["base_budget"]`). One policy then covers every point of the budget figure, instead of being evaluated on budgets it never saw. Evaluation still fixes one budget at a time on the 200-episode S4 test set. Validation during training uses the same mixed draw (it tracks the training objective).
  - **Residual decision rule (decision 2026-10-28, recorded before any residual validation result at or after 300k was looked at):**
    - *Reference line:* base-only validation success at the mixed budget, i.e. the residual env with α = 0 on the same 100 stored S4-validation episodes, with the same per-episode budget draws as the residual's validation. Logged once and drawn as a reference line on the residual validation curves.
    - *Test:* if residual SAC s0's validation success at the 300k-step evaluation is below the reference line, the residual is changed as follows and the change applies to **all residual seeds**:
      - an L2 penalty on the residual in the reward, `r ← r − w_res·‖a_res‖²` with `w_res = 0.1`, where `a_res ∈ [−1, 1]⁶` is the policy output before the α scaling (bounded, so `w_res` has a fixed meaning);
      - a critic warm-up of 50k gradient steps (actor and entropy coefficient frozen; was 20k);
      - an α ramp 0 → 0.3 over 150k env steps (was 50k).
    - *Restart point:* s0 resumes from its 100k checkpoint; on resume the warm-up and the α ramp start again from that point (the 100k actor has already been updated, so the frozen-actor phase and α = 0 restart there). Seeds 1–2 start from step 0 with the new settings.
    - *If the test passes,* nothing changes.
    - **Outcome (checked 2026-10-28 01:10):** reference = **0.88** (base plans 87%; budgets drawn 22 / 39 / 39 times at 1500 / 3000 / 6000; `results/runs/e5/base_reference.json`). Residual s0 at 300k = **0.67** → the rule triggers. All three residual seeds had already finished 1M steps on 2026-10-27 (final 0.72 / 0.68 / 0.77, all below the reference), and the 100k checkpoint had been overwritten (one rolling checkpoint per run), so the specified restart point no longer exists. **Resolution (2026-10-28):** with the reward fix (§6.7), all six E5 runs rerun from step 0, the three residual seeds with w_res 0.1, a 50k-gradient-step warm-up and a 150k-step α ramp (now the defaults in `configs/agent/residual.yaml`), after a 100k smoke test of pure SAC+HER with the fix passes (collision ≤ 0.80 and mean episode length ≥ 50 at validation; `scripts/pipeline_e5_v2.py`).
  - **Smoke result (2026-10-28 02:08): FAIL by the pre-set threshold.** Validation at 100k: collision 0.87 (> 0.80), mean episode length 153 (> 50), success 0.00. The rerun was not launched. Diagnosis on the 100 S4-validation episodes: the crash exploit is gone (v1 at the same steps: collision 1.00 after 11–14 steps; v2: collisions spread over the episode, median at step 80, 9% before step 20, the same timing as a random policy: 0.59 collision, median step 81), but the policy collides more often than random motion and has not learned avoidance by 100k (holding still collides 0%). **Decision (user, 2026-10-28): option A, launch the rerun anyway**, 2 runs at a time; started 02:17. Residual validation at 200k and 500k is logged against the base-only line (0.88 / 0.12) in `results/runs/e5/residual_milestones.log`, report only: no interventions, these runs are final.
  - **Sparse-residual arm rule (decision 2026-10-28 02:25, recorded before any 200k/500k result of the rerun was looked at):** if residual SAC s0's validation success at the 500k evaluation is **below 0.88** (the base-only line), an extra E5 arm is queued after the main six runs: **residual SAC with the sparse reward** (no HER, absorbing collision penalty, the same warm-up / α ramp / w_res as the dense residual), 3 seeds, 1M steps. It is reported **alongside** the dense residual (its own line in the budget figure and tables), never in place of it. If s0 is at or above 0.88 at 500k, no extra arm. Implementation: `configs/exp/e5_residual_sparse.yaml` (same experiment directory, `results/runs/e5`); `scripts/pipeline_e5_v2.py` applies the rule, evaluates and plots the main six first (so PE3/PE4 are not delayed), then runs the arm, evaluates it on the same 200 S4 test episodes and redraws the figure. **Update 2026-10-28 06:50:** the rule is applied as soon as residual s0 has its 500k validation; if it triggers, the arm runs on Colab after the dense seeds 1/2 (`notebooks/colab_e5_sparse.ipynb`, written by the pipeline; marker `results/runs/e5_sparse_offloaded`).
  - **α-sweep diagnostic (2026-10-28, evaluation only):** residual s0's 450k checkpoint on the 100 S4 validation episodes (same seeds and base-budget draws as training validation, deterministic actions; `scripts/alpha_sweep.py`, `results/runs/e5/residual_ff20115a_s0/alpha_sweep.json`). Success / collision at α = 0: 0.88 / 0.12 (identical to the base-only line, so the replay is exact); 0.1: 0.88 / 0.13; 0.2: 0.82 / 0.17; 0.3 (training α_max): 0.66 / 0.17 (equal to the 450k training validation). The learned residual does not improve on the base at any α and degrades it monotonically as α grows. Training-time validation uses deterministic actions at the run's current (ramped) α (`ValidationEval`).
  - **Open finding (2026-10-28): collision termination rewards crashing.** A collision ends the episode with a one-off −10, while staying alive costs −1 per step (sparse; ≈ −50 discounted at γ 0.98) or about −1 per step far from the goal (dense). Crashing early is therefore worth more than failing slowly. Pure SAC+HER on S4 learned exactly that: 0% success and 100% collisions, ending every validation episode after ~9 steps, in all three seeds (the same on 2-DOF P2: PE3 SAC+HER ends after ~15 steps, 100% collisions). The residual's validation collision rate rose from 0.12 (base) to 0.16–0.18. Fixed by the absorbing collision penalty (§6.7).
  - Base-only on the S4 test set (200 episodes, 2026-10-27): plans 77.5 / 84.5 / 90.0% at 1500 / 3000 / 6000 checks; success 79.5 / 86.0 / 90.0%, collision 20.5 / 14.0 / 8.5%. Full B2: 99.0% success, 0.5% collision (one 0.17 mm graze while tracking).
  - **Fallback when the base cannot plan** (`info["base_ok"] = False`): a straight joint-space line to the nearest collision-free IK goal (time-scaled like B2), **ignoring collisions**. The residual must steer around whatever it hits. (Day-3 first version held the start instead.)
- **Evaluation protocol for E5 (primary figure):** on the full 200-episode S4 test set (never a subset; the 30-episode S4-validation preview of 2026-10-27 was a sanity check and is not a result), for planning budgets 1500, 3000 and 6000 collision checks: base-only and base + residual (one policy per seed, trained on the mixed budgets), success and collision rate with 95% CIs; full B2 is the reference line. Secondary: jerk, minimum clearance, time to success. `scripts/evaluate.py --budgets 1500 3000 6000`.
- The base reference is replanned at every reset for that episode's goal, with an RNG drawn from the env's own seed (reproducible). Observation adds q_ref(t), q_ref(t+5) (scaled by 1/π). Validation runs at the current α.
- Residual RL is **not** used for S1/S2, where IK + quintic is already near-perfect. Those stages study pure RL (RQ1/RQ2).

- **E5 is reported as is (user decision 2026-10-29):** method and framing unchanged; a negative result with the mechanism below.
- **E5 diagnostic (evaluation only, 2026-10-29; `scripts/e5_diagnostic.py` → `results/tables/e5_diagnostic.md`; S4 test set, primary checkpoints, dense 3 seeds and sparse s1–s2).** Hypothesis (user): the base tracker re-aims `q_target` at `q_ref(t+1)` every step and so cancels the residual, bounding its authority to ≈ α·Δq_max = 0.3 × 0.04 = 0.012 rad around `q_ref`: too small to detour, enough to break the 1 cm / 5° tolerance. **Confirmed.**
  - Authority: on every step where the base action is not saturated (87% of steps), |q_target − q_ref| ≤ 0.0120 rad (max 0.0120 over 9,949 steps, dense s0, 3000 checks, 50 episodes). Larger deviations (per-episode max: median 0.027–0.041 rad, p95 ≈ 0.2 rad) occur **only** on steps where the base is saturated driving back (100% of them), i.e. transiently.
  - Outcome split (% of episodes, base-only vs dense / sparse hybrid): collisions 20.5 vs 21.2 / 20.8 (1500), 14.0 vs 15.0 / 14.2 (3000), 8.5 vs 10.0 / 9.0 (6000); **pose misses 0–0.5 vs 10–14** at every budget; timeouts 0–1 for all. The hybrid's whole loss is pose misses; it never removes a collision.
  - Final pose error on episodes where both avoided collisions (paired): base 0.41 cm / 0.88° vs hybrid 0.72 cm / 1.3° (means at 1500 and 3000), success 100% vs 84–87%.
- **E5b, pre-registered 2026-10-29 before any E5b training (user decision); reported whatever it shows.** Integrated residual offset: `o ← clip(o + α·π_res, ±0.3 rad)`, base target `q_ref(t+1) + s·o`, with `s = clip(d_pos / 5 cm, 0, 1)` so the offset fades to 0 within 5 cm of the goal (of the two options the user gave, the distance-based one: the tolerance is in task space, and a detour shifts the arm's timing against the reference). The offset persists across steps (the base no longer cancels it), and the observation adds `o / 0.3`. With π_res = 0 the env is the base exactly. Sparse reward (no HER), absorbing collision penalty, E5's warm-up / α ramp / w_res (`configs/agent/residual.yaml`), 3 seeds × 1M on S4 (`configs/exp/e5b.yaml`, `ResidualEnv(mode="offset")`). Evaluation is the same as E5: the S4 test set, budgets 1500/3000/6000, the §9.2 checkpoint rule, α sweeps at both checkpoints, its own line in the budget figure and the headline table. Tests (`tests/test_residual_offset.py`): the offset persists across steps; it reaches zero at the goal; with π_res = 0 the trajectory is identical to base-only; plus clipping and the observation. Where: seed 0 on the laptop, seeds 1–2 on Kaggle (`notebooks/colab_e5b_s1s2.ipynb`; marker `results/runs/e5b_offloaded_s1s2`; `scripts/pipeline_e5b.py`). **Offset step (user decision 2026-10-29 08:50):** `o ← clip(o + α·π_res·Δq_max, ±0.3 rad)`, i.e. at most α·Δq_max = 0.012 rad per step, so ±0.3 rad takes ~25 steps (config key `offset_step: dq_max`, run id `residual_offset_9a6a8188`). The first launch read the increment as α·π_res radians (up to 0.3 rad per step); that run (s0 to 100k, validation 0.58) is archived as `results/runs/archive/e5b_v0_rad_units/`, and E5b restarted from step 0 (laptop s0; Kaggle s1/s2 in the new kernel `colab-e5b-v1-s1s2`; the v0 Kaggle kernel `colab-e5b-s1s2` is superseded). Test added: the offset changes by at most α·Δq_max per step.
- **E5b v2 (recorded 2026-10-29 before launch; user rule applied to the v1 diagnostic).** v1 (`residual_offset_9a6a8188`, distance-only fade, w_res on π_res; s0 to 725k on the laptop, s1/s2 completed on Kaggle) was diagnosed on the 100 S4 validation episodes. **300k checkpoint** (the first kept after the warm-up; no 250k model is kept): success 0/100, collision 0.54. 95% of failures end with fade = 1 **and** |o| ≥ 0.15 rad (final |o| median 0.300 rad, the clip). At the collision step |o| is 0.299 rad (median), and ≥ 0.15 rad in 94% of collisions. Failures end 24 cm from the goal (median). The 700k checkpoint is the same: 0.05 success, 97% of failures with fade = 1 and a large offset. The Kaggle seeds finished v1 at 1M with 0.04 (s1) and 0.02 (s2) validation success. The rule's first branch applies, so v1 is stopped and archived (`results/runs/archive/e5b_v1_dist_fade/`, with `diagnostic_v1.json`). **v2** changes only:
  - (c) fade `s = min(clip(d_pos / 5 cm, 0, 1), clip(steps left in the reference / (0.1·length), 0, 1))`: also linear over the last 10% of the reference, so the target returns to the true reference by its end whatever the tool's distance;
  - w_res charges the offset, `w_res·‖o/0.3‖²` per step, instead of `w_res·‖π_res‖²`; the absorbing collision bound is unchanged (6·w_res).
  - Everything else is as v1: offset step α·π_res·Δq_max, ±0.3 rad, sparse reward, E5's warm-up / ramp / w_res value, 3 seeds × 1M, the same evaluation. Config keys `offset_fade_ref: 0.1`, `penalty: offset`; run id `residual_offset_4f49c707`.
  - Tests: the fade reaches 0 by the end of the reference even when the tool is far from the goal; the penalty scales with |o| (`tests/test_residual_offset.py`). PE4b uses v2 (companion SPEC).
  - **v2 is the final E5b iteration (user decision 2026-10-29).** It is reported whatever it shows; there is no v3.
  - **E5b v2 result on the S4 test set (2026-10-30 00:32; 3 seeds × 200 episodes; `results/tables/e5.md`, `results/figures/e5_budget_s4_test.*`):** below base-only and below the E5 hybrids at every budget, with **collisions doubled to tripled**. Success (primary / final): 1500 57.2 / 59.3, 3000 58.3 / 62.0, 6000 60.0 / 63.8 (base-only 79.5 / 86.0 / 90.0). Collision (primary / final): 42.8 / 40.7, 41.7 / 38.0, 39.8 / 36.2 (base-only 20.5 / 14.0 / 8.5). Unlike E5, the final checkpoint beats the primary everywhere. Primary checkpoints: 400k, 600k, 800k (s0, s1, s2). α sweep on validation: success falls from 0.88 at α = 0 to 0.70–0.76 already at α = 0.1 (primary) and to 0.60–0.72 at α = 0.3. α scales only the per-step increment, so a saturated offset is reached more slowly but still reached. Diagnostic at 1M (validation, s0 / s1 / s2): success 0.63 / 0.60 / 0.71, collision 0.37 / 0.40 / 0.29; 88–95% of collisions with |o| ≥ 0.15 rad, and median |o| at the last step 0.30 rad in failures and successes alike. The offset still saturates; with the authority to leave the reference, the policy drives the arm into obstacles the base avoided.
- **UR5e experiments frozen (user decision 2026-10-30):** no further E5 work. E5 and E5b are reported together as the authority–safety dilemma (`docs/thesis_notes.md`); gated and safety-filtered residuals, and TD3 for integrated actions, are future work.

### 7.4 Demonstrations & BC (RQ4, Phase 2)
- Generate N = 20k successful B2 trajectories per stage, recorded as (obs, action) at 25 Hz.
- Pre-train the actor with behavior cloning (MSE, 20 epochs), seed the replay buffer with the demos, then fine-tune with RL.
- Ablations: no demos / BC init only / BC init + demo replay.

## 8. Experiments Matrix

IDs are in numeric order; the Phase column shows which experiments run in Week 1.

| ID | Stage | Compares | Answers | Phase |
|---|---|---|---|---|
| E0 | S0 | DDPG vs TD3 vs SAC (+HER) on panda-gym | Pipeline sanity | Week 1 (optional) |
| E1 | S1 | TD3+HER vs SAC+HER (DDPG/PPO if time) | RQ1 | **Week 1** |
| E2 | S2 | Best of E1 × {sparse+HER, dense} with 6D rotation; quat variant if time | RQ1, RQ2 | **Week 1** |
| E3 | S2 | Tolerance curriculum on/off | Design choice | Phase 2 |
| E4 | S3 | Pure RL vs B2 vs residual (fixed layout) | RQ3 | Phase 2 |
| E5 | S4 | **Base + residual SAC (dense, no HER, trained on budgets drawn from {1500, 3000, 6000} checks)** vs base-only, evaluated at 1500 / 3000 / 6000 checks, vs full B2 (reference) vs pure SAC (+HER, decomposed reward) | RQ3 (headline) | **Week 1** |
| E5b | S4 | Pre-registered second iteration of E5 (2026-10-29, §7.3): base + **integrated-offset** residual SAC (sparse, no HER), 3 seeds × 1M, same evaluation as E5 | RQ3 | **Week 1** |
| E6 | S4 | Obstacle features: slots / distances / both | Design choice | Phase 2 (2-DOF PE5 answers this first) |
| E7 | S5 | Residual RL vs B3 (replanning) | RQ3 dynamic | Phase 2 |
| E8 | S4 | OOD test set + robustness with sim-to-real hooks on (eval only) | Generalization | **Week 1** |
| E9 | S4 | Residual + demos (RQ4); residual + path-restricted HER | RQ4 | Phase 2 |
| E10 | S6 | Best sim policy on URSim, then the real UR5e | RQ5 | Phase 2 |

Seeds: 3 per configuration in Week 1, except E2 (2 seeds, §11).

## 9. Evaluation Protocol

### 9.1 Metrics

**Primary (E5): success rate and collision rate**, reported against planning budget. **Secondary:** jerk, minimum clearance, time to success. The remaining metrics are reported for completeness.

| Metric | Definition |
|---|---|
| Success rate | Fraction of episodes meeting final tolerances within the horizon |
| Final position error | `d_pos` at episode end (cm) |
| Final orientation error | `θ` at episode end (deg) |
| Collision rate | Episodes with any robot/gripper–obstacle, self or table contact |
| Time to success | Steps × 0.04 s |
| Joint path length | `Σ_t ‖q_{t+1} − q_t‖₁` |
| Cartesian path length | `Σ_t ‖p_{t+1} − p_t‖₂` (TCP) |
| Smoothness (jerk) | Mean `‖d³q/dt³‖²` via finite differences at 25 Hz |
| Min clearance | Minimum over trajectory of the min body–obstacle distance (links + gripper) |
| Decision latency | Policy inference time per step; for planners, time to the first valid plan |
| Sample efficiency | Env steps to reach 80% success on the validation set |

### 9.2 Test Sets
- Generated once by `scripts/make_testsets.py` with fixed seeds and stored as `.npz`.
  - Week 1: S1, S2 and S4 with 200 episodes each, plus a 100-episode S4 OOD set (more and larger obstacles, boundary goals).
  - Phase 2: expand to 500 episodes per stage.
- Each episode stores q₀, the goal pose and the obstacle layout.
- Never used for training or tuning. A separate 100-episode validation set per stage is used for curriculum and model selection.
- **Evaluation checkpoint rule for residual arms (decision 2026-10-28 07:20, recorded before any test-set evaluation; every residual arm, dense and sparse):**
  - **Primary:** the best-validation checkpoint (success, then lower final position error) among the kept checkpoints (every 100k) taken **after the α ramp has completed**. Implemented as after the α ramp *and* the critic warm-up: with the §7.3 settings the ramp ends at 150k env steps but the actor stays frozen until the 50k-th gradient step (≈ 202k env steps at 0.25 per transition after 2k learning starts), so the 200k checkpoint is still the untrained near-zero residual, i.e. the base alone. First eligible checkpoint: 300k. (`scripts/checkpoints.py`; this interpretation is flagged for the user.)
  - **Secondary:** the final 1M checkpoint.
  - **Analysis figure:** the validation α sweep {0, 0.1, 0.2, 0.3} at both checkpoints (`scripts/alpha_sweep.py --checkpoint primary|final`, `scripts/plot_alpha_sweep.py` → `results/figures/e5_alpha_sweep.*`).
  - The budget figure draws each residual arm at the primary checkpoint (solid) and the final one (dashed); test-set keys are `hybrid@N:<run>#primary|final`. Non-residual policies keep the plain best-validation checkpoint (`best_model.zip`).
- **Built 2026-10-27** (`testsets/`, committed; `testsets/summary.json`):

  | Set | Episodes | Seeds from | Rejected (no collision-free path) | B2 success | B2 duration p50 / p99 |
  |---|---|---|---|---|---|
  | S1 / S2 test | 200 / 200 | 3.0M / 3.1M | 0 | — | — |
  | S4 test | 200 | 3.2M | 19 | 99.5% | 6.16 / 13.00 s |
  | S4 OOD test (4 spheres r 8–14 cm, goals ≥ 0.75 m from the shoulder) | 100 | 3.3M | 5 | 98% | 6.86 / 14.67 s |
  | S1 / S2 / S4 validation | 100 each | 1.0M / 1.1M / 1.2M | 0 / 0 / 4 | S4: 100% | S4: 6.16 / 13.57 s |

  - **Disjointness (checked 2026-10-27):** all seven sets are pairwise disjoint by seed range and by episode content (start + goal configuration). Validation never replays a test set. Validation seed ranges differ per stage, because S1 and S2 sample identically (the first build had S1/S2 validation share 1.0M seeds and therefore the same episodes; S2 and S4 validation were regenerated).

  - S4 feasibility: B2 with a 2 s budget, else RRT with 10 s; an episode with neither is replaced by the next seed.
  - Training validation replays the stored validation set (`reset(options={"episode": ...})`), so S4 validation contains no infeasible episode. For S1/S2 the stored and seed-regenerated sets are identical (checked).

### 9.3 Statistics
- Report mean ± 95% bootstrap confidence intervals across seeds.
- For learning curves, report IQM and performance profiles (the `rliable` library; Agarwal et al., 2021).
- Never report single best seeds.
- **Seed-count caveat (stated once in the results chapter):** with 2–3 seeds, seed-level confidence intervals are close to uninformative. As a secondary, clearly labeled measure, report success rates with a Wilson interval pooled over all test episodes (e.g. 3 seeds × 200 episodes). This captures episode-level variation, **not** seed-to-seed variation.

## 10. Visualization

**Local interactive viewer** (`scripts/play.py`, on the laptop)
- Loads a checkpoint and runs the policy in `mujoco.viewer.launch_passive`.
- The goal is a **mocap body** you can drag in the viewer; the policy reads its pose every step, so you can steer the arm live.
- Overlays drawn through `viewer.user_scn`: goal and TCP coordinate frames, TCP trail, obstacle safety-margin shells, and the base reference path (residual mode).
- Keys: `R` reset, `N` new random goal / scene, `P` pause, `B` toggle base-only vs residual.
- **Replay mode (built 2026-10-30):** `--testset s4_test --episode i --method base|hybrid|b2 [--budget N] [--run <residual run> --checkpoint primary|final]` replays one stored test episode exactly as `scripts/evaluate.py` runs it: same env config, reset seed, base plan, and B2's shared planning rng. Checked on episodes 32 and 8 for all four methods, where outcomes match the evaluation rows, including the 1.1 cm miss. `--record <file.mp4>` writes an offscreen video (960×720, 25 fps = real time; `imageio` + `imageio-ffmpeg`, viewer-only dependencies). In replay mode, `N` steps to the next test episode, and `B` switches the correction off (α = 0, offset cleared) from that step on. Videos: `results/videos/ep{32,8}_{1_base3000,2_e5_hybrid_s0_primary,3_e5b_hybrid_s0_final,4_b2}.mp4`.

**Colab renderer** (`ur5rl/viz/render.py`)
- Set `MUJOCO_GL=egl`, render offscreen with `mujoco.Renderer`, write MP4 with `mediapy`.
- Produces a grid video of N evaluation episodes per method for side-by-side comparison in the thesis and presentation.

**Plots:** learning curves, success vs tolerance, error histograms, jerk/clearance box plots, and per-joint trajectories against the B1/B2 baselines.

## 11. Compute Plan (1 week, CPU only)

| Experiment | Runs | Steps / run (default) | Total steps |
|---|---|---|---|
| E1 | 2 algorithms × 3 seeds | 300k | 1.8M |
| E2 | 2 reward modes × 2 seeds | 1M | 4M |
| E5 | pure SAC + residual SAC × 3 seeds | 1M | 6M |
| **Total** | | | **≈ 12M** |

- **Throughput gate (hard): total steps/s per machine ≥ 100**, summed over the runs that machine executes in parallel (S4 env, SAC end to end, 10k steps, gradient updates running). Benchmark every machine type with `scripts/bench_throughput.py` before launching anything on it. Step-time profiling (§6.5) is reported alongside as a diagnostic.
- **Day-1 measurements, laptop (i5-1240P, 4P+8E cores, 15.7 GB RAM).** The SAC gradient step dominates (26–36 ms per step for 3×256 at batch 256); the env step is ~0.9 ms, of which physics is 0.24 ms.

  | Network | Grad steps / transition | Parallel runs | Steps/s per run | Total |
  |---|---|---|---|---|
  | 3×256 | 0.5 | 1 | 66 | 66 |
  | 3×256 | 0.25 | 1 | 119 (HER 114) | 119 |
  | 3×256 | **0.25** | **4** | **54** | **216** |
  | 2×256 | 0.5 | 1 | 91 | 91 |
  | 2×256 | 0.5 | 4 | 30.6 | 122.5 |
  | 2×256 | 0.25 | 1 | 167 (HER 150) | 167 |

  Torch threads: 2 per run is best (1: 42, 8: slower). SubprocVecEnv gains nothing. Frame skip 10 at 0.004 s saves 0.12 ms per step (~1%), so it is **not** a useful fallback; it is dropped from the gate-fail path.
- **Budget (3×256 at 0.25, laptop alone, 216 steps/s total):** the 12M-step plan is ~15–16 laptop-hours: E1 ~2.5 h, E2 ~5 h, E5 ~10–11 h plus RRT planning time (not yet measured; a 0.2 s plan per episode adds up to ~10%). Colab/Kaggle are extra capacity, not required.
- **Schedule after Day 3 (decisions 2026-10-27, revised the same night):**
  - **E5 does not depend on E1** (it trains SAC either way), so it starts in the laptop slots E1's first wave frees, without waiting for E1. Rule: at most 3 UR5e training jobs alive on the laptop (`launch.py --cap-ur5e 3`), each start gated on 1.7 GB free RAM. E5 started 03:53 (2 runs) and took the third slot at 04:39 when E1 SAC s0 finished.
  - **E1's second wave (TD3 s2, SAC s1, SAC s2) moves to Colab/Kaggle** (`notebooks/colab_e1_wave2.ipynb`, one run at a time, ~80 min per run on Colab), or runs on the laptop after E5 if it is still missing then (create `results/runs/e1_offloaded` while the cloud is still running it). TD3 s2 and SAC s1 had just been started by the old launcher; they were stopped before their first checkpoint.
  - **E1 breakdown, E2 algorithm pick and E2** happen once all six E1 runs exist, wherever they ran. E2 then goes to Colab/Kaggle (`notebooks/colab_e2.ipynb`, marker `results/runs/e2_offloaded`) or runs on the laptop after E5.
  - **2-DOF lane:** at most 1 worker while any E5 job runs (`launch.py --yield-to e5.yaml --yield-workers 1`), 2 otherwise; each start also needs 2.5 GB free RAM. PE3/PE4 run 5 seeds (companion spec minimum), extended to 10 later only if time remains.
  - **RAM cap (decision 2026-10-27 04:53):** RAM cannot be freed, so at most **2** UR5e training jobs run on the laptop (`--cap-ur5e 2`); each E5 run sits at ~1.2 GB right after resuming (the checkpointed 1M-transition buffer loads at full size) and ~1.7 GB at the end. The third E5 job (SAC+HER s1) was stopped at ~27k steps, before its first checkpoint, and restarts from 0 as soon as either running E5 job finishes (it is next in the launcher queue). The 2-DOF lane starts a job only with ≥ 1.0 GB free, which leaves room for exactly one ~0.35 GB job beside the two E5 runs.
  - **Incident (04:51):** every training job and both pipelines stopped at once without a traceback (`^C` in the pipeline output; no reboot, no out-of-memory event). **Cause: the IDE was closed, and a launcher was running in its terminal**, so the interrupt reached the whole process tree. E5 SAC+HER s0 and residual s0 resumed from their 100k checkpoints (≤ 25k steps lost each); 2-DOF jobs restarted. Since then every job starts in its own process group with Ctrl+C disabled (`CREATE_NEW_PROCESS_GROUP` in `launch.py` and both pipelines).
  - **Incident 2 (2026-10-28 06:17):** all jobs (E5 SAC+HER s0 and residual s0, both near 455k; PE8 SAC+HER s1 at ~80k) and both pipelines stopped again with `^C`, no traceback, no reboot or out-of-memory event: a console close reaches every process attached to that console, whatever its process group. Fix: every child is now started with `DETACHED_PROCESS` as well (no console). Both E5 runs resume from their 450k checkpoints (≤ 10k steps lost each), PE8 s1 from its last checkpoint.
  - **Checkpoint retention (2026-10-28):** the rolling checkpoint (model + replay buffer, every 50k) is still overwritten for resume. Kept and never overwritten: the model alone at every 100k (`checkpoints/model_<steps>.zip`, ~7 MB) and model + buffer at 100k and 300k (`checkpoints/full_<steps>/`), the §7.3 restart points. Full copies every 100k would need ~51 GB for E5 (0.85 GB buffer each); the disk has ~24 GB free.
  - **2-DOF lane yields to every UR5e job** (`--yield-to ur5e`), not only E5: at most 1 worker while any UR5e training job runs (it ran 2 beside E2 and pushed E2 into paging). Since 2026-10-28 06:36 a UR5e launcher that is still waiting for RAM counts too, so the 2-DOF lane cannot take the memory a pending E5 start needs.
  - **E5 rerun gets both laptop slots (2026-10-28 02:13):** E2 seed 1 was paused (sparse+HER s1 at its 50k checkpoint; dense s1 had just started and restarts from 0). The E2 seed-1 runs follow E5 and its evaluation (`scripts/pipeline_e5_v2.py`), unless `results/runs/e2_offloaded` exists (Colab). Order on the laptop: E5 (6 runs, 2 at a time) → S4-test evaluation and budget figure (`results/runs/e5/EVAL_DONE`) → E2 seed 1. 2-DOF lane: PE8 until EVAL_DONE, then PE3 → PE4 → rest of PE8 (1 worker while any UR5e job runs).
  - **Offloading E5 (decision 2026-10-28 06:46; Chrome stays open, so RAM is short):** the laptop runs only E5 seed 0, one job at a time (residual s0 from its 450k checkpoint, then SAC+HER s0 from 450k); seeds 1 and 2 (residual and SAC+HER) run on Colab one at a time (`notebooks/colab_e5_s1s2.ipynb`, from `scripts/make_colab_notebook.py`; marker `results/runs/e5_offloaded_s1s2`; run ids identical to the laptop's). Colab runs checkpoint every 50k and keep no extra model + buffer copies (`train.py --keep-full-at` with no steps), only to save Drive space; no training setting changes. RAM gate for a laptop E5 start: **1.4 GB** available, from the measured peak RSS of the six completed 1M-step v1 E5 runs (1.18–1.30 GB). Residual s0 resumed at 1.23 GB working set. **2-DOF lane fully paused** (`--yield-workers 0`) while a UR5e job runs or a UR5e launcher waits; it resumes when the laptop's seed-0 runs finish.
  - **Laptop order (decision 2026-10-28 07:20):** residual s0 → SAC+HER s0 → sparse-residual s0 → 2-DOF lane (PE8, then PE3/PE4 after the E5 evaluation). Offloaded: dense residual and SAC+HER s1/s2 (`notebooks/colab_e5_s1s2.ipynb`, residual s1, residual s2, SAC+HER s1, SAC+HER s2) and sparse residual s1/s2 (`notebooks/colab_e5_sparse.ipynb`, Colab or Kaggle; markers `results/runs/e5_offloaded_s1s2`, `results/runs/e5_sparse_offloaded_s1s2`). `results/runs/UR5E_LANE_BUSY` (the pipeline's pid) keeps the 2-DOF lane paused between the laptop's UR5e launches.
  - **Windows power throttling (found 2026-10-28 07:17):** a console-less training job was put into efficiency mode (EcoQoS) a few minutes after start and slowed ~3× at unchanged workload (residual s0: 110 → 36 steps/s, ~12 page faults/s, so not paging). Disabling power throttling for the process restored 159 steps/s. `train.py` now opts out at startup (`no_power_throttling`); speed only, no effect on training. E5 evaluation waits until all six dense runs are in `results/runs/e5`.
  - **E2 shortfall (1-week rule), 2 seeds:** S2 target ≥ 90% success at 1 cm / 5°. Final validation at 500k steps (100 episodes; table: `results/tables/e2.md`, from `scripts/e2_table.py`):
    - SAC sparse+HER: **4.5%** mean success (s0 3%, s1 6%); position error 35.4 cm (31.1 / 39.7), orientation error 67.6° (61.7 / 73.6).
    - SAC dense: **0%** (s0 0%, s1 0%); position error 30.2 cm (31.3 / 29.1), orientation error 54.2° (53.5 / 54.9).
    - Seed 1 of both ran on Colab (`notebooks/colab_e2.ipynb`, identical config hashes `a337c4b7` / `55944f05`) and was merged into `results/runs/e2/`; the laptop's paused partial s1 folders are in `results/runs/archive/e2_s1_laptop_partial/`.
    - **Contact in S2 is free:** S2 neither penalises nor terminates on contact (§6.8), and the policies use it. Table/self-contact rate at the final validation: sparse+HER 30% (s0) and 47% (s1), 28–55% over the last 100k steps; dense 12% / 9%. The S2 success numbers are therefore not contact-free successes, and the S2 policies are not a collision-aware starting point for S3/S4.
  - **E5 complete (2026-10-29 06:00):** all nine runs at 1M with `done.json` (dense residual s0–s2, SAC+HER s0–s2, sparse residual s1–s2), except sparse residual s0 (laptop, resumed at 200k). Dense s1/s2 ran on a Kaggle TPU-VM CPU host, four jobs in parallel at 1 torch thread each (2.2 h per run); sparse s1/s2 on Kaggle (3.1 / 3.3 h). Config hashes verified on merge (`ff20115a`, `e2297f53`, `152018f8`); nothing was overwritten. Thread count is not part of the config, so the run ids match the laptop's.
  - **Evaluation before sparse s0 (decision 2026-10-29):** the dense E5 evaluation (§9.2 rule: hybrid@N at the primary and final checkpoint, pure SAC+HER, α sweeps at both checkpoints) runs first, in two parallel streams (`evaluate.py` locks `summary.json`), then the rest of sparse s0 (it needs 1.4 GB, which Chrome leaves only when nothing else runs), then the sparse evaluation. Headline table: `scripts/e5_table.py` → `results/tables/e5.md` (§9.3: mean over seeds with a 95% bootstrap CI over seeds, plus the pooled Wilson interval as the labelled secondary measure).
  - **Incidents 3 and 4 (2026-10-28 10:02, 2026-10-29 02:39):** all jobs and both pipelines died again at once, no traceback, and no crash, sleep, reboot or out-of-memory event in the System or Application log. `DETACHED_PROCESS` did not prevent it. Every venv Python runs inside a kill-on-close job object (the venv launcher's, flags `0x3000`), so killing one launcher process takes its interpreter with it; what triggers the kill is outside the code and still unknown. **Mitigation:** `scripts/watchdog.py` restarts any pipeline that is not running and not finished (markers `results/runs/e5/PIPELINE_DONE`, `results/runs/PIPELINE_2DOF_DONE`), logging each restart to `results/runs/watchdog.log`. Registering it as a 10-minute scheduled task (`scripts/install_watchdog.ps1`) needs the user's go-ahead; until then it is run by hand.
  - **Launcher fix (2026-10-29):** `launch.py` drops runs that already have `done.json` before any RAM gating. A relaunched E5 pipeline's launcher had waited for 1.4 GB to start two finished runs, and the 2-DOF lane yielded to that waiting launcher, so both lanes stalled.
  - **2-DOF offload (decision 2026-10-29):** PE3, PE4 and PE8 seeds 2–4 run on Kaggle (`notebooks/colab_pe_s234.ipynb`, jobs in parallel, one per CPU core, 1 torch thread each; markers `results/runs/pe34_offloaded_s234`, `pe8_offloaded_s234`), seeds 0–1 on the laptop, which also carries the E5 evaluation and sparse s0. That is 3 of 5 seeds offloaded rather than half, because the laptop's RAM is the bottleneck. The bundle now contains `2-dof/`.
  - **E5 result on the S4 test set (2026-10-29 07:38; dense 3 seeds, sparse s1–s2, sparse s0 pending; `results/tables/e5.md`, `results/figures/e5_budget_s4_test.*`, `e5_alpha_sweep.*`):** the hybrid is **below base-only at every budget**. Success, base-only vs dense hybrid (primary / final): 1500: 79.5 vs 66.2 / 61.5; 3000: 86.0 vs 71.5 / 66.5; 6000: 90.0 vs 74.8 / 69.7. Sparse hybrid (primary / final): 69.2 / 59.0, 74.2 / 63.3, 77.5 / 66.0. Collision is at the base's level for every arm (≈ 21 / 15 / 10%). Full B2: 99.0% / 0.5%. Pure SAC+HER: 0% success, 53.8% collision. The §4 S4 criterion ("base + residual at ≤ 6000 checks matches full B2, where base-only does not") is **not met**: the residual makes the base worse. α sweep: validation success falls monotonically from 0.88 at α = 0 on every seed and checkpoint. The primary rule picked 600k–800k for the dense seeds and 300k (the first eligible checkpoint) for both sparse seeds: the best residual is the least-trained one.
  - **2-DOF lane beside UR5e jobs again (2026-10-29 07:40):** `pipeline_2dof.py` runs 2 workers (1.0 GB RAM gate) while a UR5e job trains. The full pause existed for pending UR5e starts; with sparse s0 already running and 3.3 GB free, it only idled the laptop, and the user asked for PE3/PE4 in parallel.
  - **Disk full (2026-10-29 19:13):** drive D: (49 GB) filled up; E5b v2 s0 crashed copying its 300k model + buffer snapshot (`[Errno 28]`). Its rolling 300k checkpoint was intact (buffer 900 MB, byte-identical to the rebuilt `full_300000`), and the run resumed from 300k at 19:22. To free space without deleting anything, two archived, unreported folders were **moved** to `C:\mtp_offload
esults_runs\`: `e5_v1_crash_exploit` (4.8 GB) and `archive/e5b_v0_rad_units` (1.8 GB). Only comments refer to them. Reclaimable with the user's approval: the rolling replay buffers of 130 finished runs (10.9 GB, needed only to resume, and a finished run never resumes) and the `full_*` snapshots (7.0 GB, the §7.3 restart points).
  - Scripts: `scripts/pipeline_e1_e5.py` (UR5e lane) and `2-dof/scripts/pipeline_2dof.py` (PE1 → PE2 → PE3 → PE4). `launch.py` never starts a job that is already running on the machine.
- **RAM per run (measured):** ~0.47 GB baseline (torch + 5 lean envs) + 604 B per buffer transition + a transient copy of the buffer while a checkpoint is pickled. E1 (300k): ~0.8 GB peak per run. 1M-step runs (E2, E5): ~1.6 GB peak per run. **Parallel runs per machine are limited by free RAM**, not only cores: at 2.5 GB free on Day 1 the laptop runs 3 E1 jobs; 4 need ~3.3 GB free, and four 1M-step runs need ~6.5 GB.
- **If the gate fails on a machine:** run fewer, longer jobs there (the gate is a total), then cut in this order: E2 to 1M → 600k steps; E5 to 800k steps; E1 to TD3 vs SAC with 2 seeds.
- **Measured numbers are reported before any training launches,** and the week's step budgets are set from them.
- No GPU required. MJX/GPU is Phase 2 only.
- **Resilience:**
  - Checkpoint plus replay buffer saved to Drive every 50k steps.
  - Training resumes from the last checkpoint automatically.
  - Each run ID encodes config hash + seed.

## 12. Repository Structure

```
ur5e-rl-planner/
├── SPEC.md
├── README.md
├── pyproject.toml              # pinned deps (lockfile committed)
├── configs/
│   ├── env/  s1.yaml s2.yaml s3.yaml s4.yaml s5.yaml
│   ├── agent/ td3.yaml sac.yaml ddpg.yaml ppo.yaml residual.yaml
│   └── exp/  e0.yaml … e10.yaml # composes env + agent + seeds
├── assets/
│   └── scene_ur5e.xml          # menagerie UR5e + Robotiq 2F-85, table, mocap goal, obstacle slots
├── src/ur5rl/
│   ├── envs/      base.py pose_env.py obstacle_env.py sampling.py rotations.py randomization.py
│   ├── planners/  ik.py quintic.py rrt_connect.py smoothing.py time_param.py
│   ├── agents/    factory.py residual.py demos.py bc.py callbacks.py
│   ├── eval/      metrics.py evaluate.py testsets.py stats.py
│   └── viz/       viewer.py render.py overlays.py plots.py
├── scripts/       train.py evaluate.py play.py make_testsets.py gen_demos.py
├── notebooks/     00_pandagym_replication.ipynb  01_colab_train.ipynb  02_results.ipynb
├── tests/         test_fk_tcp.py test_rotations.py test_reward_her.py
│                  test_env_api.py test_sampling.py test_planners.py
├── arm2dof/       # companion 2-DOF study (own SPEC.md)
└── results/       # gitignored; runs/<exp>/<seed>/
```

**Stack:** Python 3.11, `mujoco`, `gymnasium`, `stable-baselines3`, `numpy`, `scipy`, `mediapy`, `tensorboard` (or `wandb`), `rliable`, `pytest`. Optional: `mink`. Phase 2: `ur_rtde`, URSim, `mujoco-mjx`, `jax`.

## 13. Testing (must pass before any training run)
- **FK:** TCP pose matches known UR5e + gripper poses (home, zero) to < 1e-6; TCP offset matches the model site.
- **Rotations:** quat ↔ matrix ↔ 6D round-trip; geodesic angle properties (θ(R,R)=0, symmetry, π at 180°).
- **`compute_reward`:**
  - batch result equals per-sample result;
  - matches `step()` reward exactly;
  - the HER relabel path preserves collision, margin and smoothness penalties (§6.7): rewards from `SafetyHerReplayBuffer`'s virtual samples equal `compute_reward` with the original infos, bit for bit, and differ from the rewards without the safety terms;
  - SB3's stock `HerReplayBuffer` (empty infos) makes `compute_reward` raise instead of silently dropping the terms;
  - the HER buffer pickles (checkpoint/resume).
- **Stage rules:** in S1/S2 a contact neither terminates the episode nor adds a penalty (§6.8).
- **Goal independence (§6.4):** `observation` is bit-identical when the goal is changed (with and without obstacles), and the policy's goal-error features equal `desired_goal − achieved_goal` for a relabeled goal.
- **Planners (Day 3, `tests/test_planners.py`):** analytic IK returns the generating configuration and every solution satisfies FK within 1e-4 (500 random poses; observed ~1e-11); quintic starts/ends at rest within v/a limits; B2 paths collision-free at a 10× denser re-check (0.001 rad); B1 tracked through the env reaches its goal.
- **Residual mode:**
  - the config loader rejects `residual: true` with `her: true`;
  - `q_ref` in every stored transition was planned for that transition's `desired_goal`;
  - with the residual forced to zero, the agent's actions equal `a_base` exactly, and the reward equals the dense formula on the base-only rollout;
  - at initialization (log-std bias −3), the mean |α_max · π_res| over 1k samples is below 0.02;
  - each episode's base reference ends at that episode's goal pose (FK of the last point, 1e-6);
  - `WarmupSAC` leaves the actor bit-identical during warm-up while the critic changes, and updates it afterwards.
- **Clearance:**
  - analytic capsule–sphere distances match `mj_geomDistance` within 1e-6 on 1k random configurations, for all arm links **and** both gripper capsules;
  - every vertex of the 2F-85 visual mesh lies inside at least one gripper capsule (enclosure test);
  - step time is profiled, not asserted (§6.5 soft targets).
- **Action rule (§6.2):**
  - `q_target` integrates `a · Δq_max`, is clipped to the joint limits, and never leads q by more than 0.25 rad (300 random steps);
  - `q_target = q` after reset, and the observation's lead block equals `(q_target − q) / 0.25`;
  - realised joint speed at |a| = 1, from home in a contact-free direction, is > 0.95 · Δq_max / 0.04 s on every joint;
  - holding a target (a = 0) at sampled configurations, the arm settles within 1 mrad (gravity compensation).
- **Goal filter (§6.5):** with the filter on, every sampled goal satisfies `max_j |Δq_j| ≤ 0.8 · v_max · T` and lies within ±π.
- **Reachability (script, not a unit test):** `scripts/check_reachability.py` gives 100% oracle success among contact-free episodes for S1, S2 and S4 timing. Re-run whenever the action rule, horizon, limits or actuators change.
- **Environment API:** passes `gymnasium.utils.env_checker` and SB3 `check_env`.
- **Sampling:** every sampled goal and start is collision-free (gripper included) and within joint limits.
- **Planners:** IK solutions satisfy FK within 1e-4; RRT paths are collision-free at a dense re-check resolution.
- **Randomization hooks:** with all hooks off, results are bit-identical to the base environment.

## 14. Reproducibility & Logging
- Every run saves the resolved config, git commit, package versions, seed and hardware info.
- Logged scalars: return, success rate, pose errors, collision rate, current tolerance (curriculum), entropy coefficient, actor/critic losses, ‖residual‖ (residual runs).
- Eval runs every 25k steps on the validation set, logging the best checkpoint and the last checkpoint.

## 15. One-Week Plan

This plan covers both specs; the 2-DOF tasks are listed so the week is planned as one. Training jobs are launched as early as possible and run in the background while the next day's code is written.

| Day | UR5e (this spec) | 2-DOF (companion spec) | Background training |
|---|---|---|---|
| 1 | Repo, scene XML (UR5e + 2F-85, TCP site), FK/rotation tests, analytic clearance + step-time profile, **throughput gate (§11)** | Kinematic env, FK/IK, geometry, C-space grid + A*, tests | E0 (optional), 2-DOF PE1/PE2 |
| 2 | S1/S2 env, HER reward decomposition + tests, local viewer | P2/P3 envs, RRT-Connect, residual wrapper + **`WarmupSAC`** + residual init (§7.3), two-panel viewer | **E1**, 2-DOF PE3/PE4 |
| 3 | IK + quintic (B1), RRT-Connect (B2), frozen test sets S1/S2/S4 | P5 torque env, dynamics cross-check, computed-torque baseline | ~~E2~~ → **E5 pure + residual SAC (laptop, from the night of Day 3)**; E1 second wave on Colab/Kaggle; 2-DOF PE1 → PE2 → PE3/PE4 (1 worker) |
| 4 | S3 (debug) → S4 obstacle env; apply 2-DOF design recommendations | Evaluate PE1–PE4, PE6 C-space figures → recommendations | E5 continues; **E2** on Colab/Kaggle once E1 is complete (else laptop after E5) |
| 5 | Residual SAC on S4 | PE9-lite (compare 2-DOF vs UR5e design rankings) | E5 finishes; E5 evaluation (budget figure); E2 if still pending; PE3/PE4 seeds 5–9 if time remains |
| 6 | Evaluation on frozen sets (E1, E2, E5, E8), stats, videos | Final 2-DOF evaluation, figures | — |
| 7 | Write-up: results tables, figures, README; buffer | Chapter figures | — |

**Cut line, if behind schedule, in this order:** drop E0 → drop the quat ablation → E2 down to 600k steps → E8 down to OOD only → 2-DOF PE9-lite. **E5 (residual vs pure vs B2) is never cut**, because it is the contribution.

## 16. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| 1 week is not enough for ~12M training steps | High | Reduced update ratio and batch (§7.1); Day-1 throughput gate (§11); ≥4 parallel CPU sessions; cut line in §15 |
| HER relabeling inconsistent with the residual reference | High if ignored | Residual agents use dense reward without HER; enforced by config check and tests (§6.7, §13) |
| Residual noise destroys base behavior from step 1 | High if ignored | Log-std bias −3, low initial entropy coefficient, target entropy −12, α ramp, `WarmupSAC` (§7.3) |
| Clearance computation dominates step time | Medium | Analytic NumPy distances, spheres only in Week 1, rigid 2-capsule gripper; physics/clearance profiled on Day 1 (≤2 ms / ≤0.1 ms soft targets) |
| Physics step too slow for the throughput gate | Medium | Rigid gripper (no linkage); frame skip 10 at 0.004 s after an actuator-stability check; then cut line (§11) |
| Sparse+HER fails on tight 6D pose | High | Tolerance curriculum; dense reward |
| Pure RL can't solve S4 clutter | Medium | Expected; residual RL is the answer; report pure RL as a negative result |
| Residual policy ignores or destroys base | Medium | Near-zero init, critic warm-up, α schedule; log ‖residual‖ |
| Gripper adds collisions near goals (narrow approach) | Medium | Gripper in goal-sampling rejection; clearance metric includes gripper |
| Colab disconnects / quota limits | High | Checkpoint to Drive, auto-resume, spread seeds across Kaggle and laptop |
| Joint wrap-around / IK branch flips | Medium | Narrowed joint limits; nearest-solution IK; sin/cos plus normalized q features |
| Sim-to-real gap larger than expected (Phase 2) | Medium | Hooks in §6.9 built now; URSim before real arm; conservative speed limits |

## 17. Resolved Decisions

| # | Question | Decision |
|---|---|---|
| 1 | Real robot? | **Simulation only for now.** A real-robot stage (S6) follows after the sim work is finished. Sim is built real-robot-ready (§6.2, §6.9). |
| 2 | Novelty expectation | The guide expects **a minimum novel method**. This is defined in §3.1: planner-guided residual SAC (dense reward, no HER), HER reward decomposition for the goal-conditioned agents, validated 2-DOF → 6-DOF. |
| 3 | Timeline | **1 week** for the build and core results (§15). Everything else is Phase 2. |
| 4 | End-effector | **Gripper (Robotiq 2F-85)**, fixed opening, included in TCP and collision model (§6.1). |
| 5 | Design review (Claude Code, v0.3) | HER kept out of residual training; residual starts near zero in mean *and* noise; `WarmupSAC`; normalized q in observation; reduced update ratio + Day-1 throughput gate; analytic clearance with spheres only in Week 1. |

## 18. References

**RL foundations**
- Lillicrap et al., *Continuous control with deep RL* (DDPG), 2015 — arXiv:1509.02971
- Andrychowicz et al., *Hindsight Experience Replay*, 2017 — arXiv:1707.01495
- Fujimoto et al., *Addressing Function Approximation Error in Actor-Critic Methods* (TD3), 2018 — arXiv:1802.09477
- Haarnoja et al., *Soft Actor-Critic*, 2018 — arXiv:1801.01290
- Schulman et al., *Proximal Policy Optimization*, 2017 — arXiv:1707.06347

**Representation & evaluation**
- Zhou et al., *On the Continuity of Rotation Representations in Neural Networks*, CVPR 2019 — arXiv:1812.07035
- Agarwal et al., *Deep RL at the Edge of the Statistical Precipice* (rliable), 2021 — arXiv:2108.13264

**Residual RL**
- Silver et al., *Residual Policy Learning*, 2018 — arXiv:1812.06298
- Johannink et al., *Residual Reinforcement Learning for Robot Control*, 2018 — arXiv:1812.03201

**RL & neural motion planning for manipulators**
- *URPlanner: A Universal Paradigm for Collision-Free Robotic Motion Planning Based on DRL*, 2025 — arXiv:2505.20175
- *DRL-based Path Planning for Industrial Robots using Distance Sensors as Observation* (UR5), 2023 — arXiv:2301.05980
- Fishman et al., *Motion Policy Networks*, CoRL 2022 — arXiv:2210.12209
- Qureshi et al., *Motion Planning Networks*, 2018 — arXiv:1806.05767
- Sundaralingam et al., *cuRobo: Parallelized Collision-Free Minimum-Jerk Robot Motion Generation*, 2023 — arXiv:2310.17274

**Tooling**
- Zakka et al., *MuJoCo Playground*, 2025 — arXiv:2502.08844
- MuJoCo Menagerie — `google-deepmind/mujoco_menagerie` (UR5e and Robotiq 2F-85 models)
- `ur_rtde` (real-time UR control interface) and URSim — Phase 2
