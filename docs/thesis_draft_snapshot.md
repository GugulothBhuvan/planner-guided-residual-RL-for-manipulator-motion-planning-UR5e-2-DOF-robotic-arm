# ResPlan: Residual RL over Cheap Motion Planners for a 6-DOF Arm

Oct 30, 2026 · @Bhuvan Raj Guguloth

MTP thesis draft, UR5e chapters. Numbers are from the frozen experiments in the repo (SPEC v0.24); the 2-DOF chapter is filled once its evaluation finishes.

## Abstract

We test whether a cheap sampling-based planner plus a learned residual correction can match an expensive planner for collision-free 6D pose reaching on a simulated UR5e with a Robotiq 2F-85 gripper. It cannot. On 200 held-out cluttered scenes, RRT-Connect with a full budget succeeds in 99.0% of episodes. With 1,500 to 6,000 collision checks it succeeds in 79.5% to 90.0%, and adding a learned SAC residual lowers success at every budget.

Two residual designs fail in opposite ways. An additive residual, applied on top of a tracking controller, is cancelled every step and can move the arm at most 0.012 rad from the planned path: too little to avoid obstacles, enough to miss a 1 cm goal tolerance. An accumulated offset has the authority to steer, but the policy drives it to its limit and collisions rise from 8.5–20.5% to 40–43%. We call this the authority–safety dilemma of residual correction over sampling planners.

Along the way we document three failure modes of goal-conditioned training with hindsight relabeling (HER) that silently broke learning: goal-dependent observation features, safety penalties dropped by the replay buffer, and a collision termination that made crashing the best-paying behaviour. Each comes with a reusable fix. A planar 2-DOF study validates the methodology (Chapter 11).

## 1 Introduction

Sampling-based planners such as RRT-Connect find collision-free paths reliably when given enough computation, but their success falls as the planning budget shrinks. Residual reinforcement learning promises a way around this: keep a cheap classical controller for safety and let a learned policy add small corrections where it falls short (Silver et al., 2018; Johannink et al., 2019). This thesis tests that promise on a concrete case: 6D pose reaching with a UR5e arm and gripper among randomly placed obstacles.

The study is simulation-only and was built and run in one week on CPU hardware. Every design decision, deviation and shortfall was recorded in a versioned specification (SPEC.md) before the result it affects was seen.

**Research questions**

1. RQ1: Which off-policy algorithm learns 6D-pose reaching best under a fixed step budget (SAC vs TD3, with HER)?
2. RQ2: How do sparse rewards with HER compare to dense shaped rewards?
3. RQ3: Can a learned residual on top of a cheap RRT-Connect planner match the full planner's success and collision rate?
4. RQ5 (deferred): how do results transfer to the real arm? This is out of scope here.

**Contributions**

- A reproducible MuJoCo benchmark for the UR5e with a gripper: frozen test sets, planner baselines at three budgets, and a pure-RL reference.
- A diagnosed negative result for RQ3: two residual designs, the measured cause of each failure, and the authority–safety dilemma they bracket.
- Three reusable findings on goal-conditioned training with HER, each with a tested fix.
- A 2-DOF companion study that validates the methodology where exact best-known paths exist.

## 2 Background and related work

**Off-policy actor-critic RL.** DDPG (Lillicrap et al., 2015) learns a deterministic policy from replayed transitions. TD3 (Fujimoto et al., 2018) adds twin critics and delayed actor updates to curb overestimation. SAC (Haarnoja et al., 2018) adds an entropy bonus that keeps the policy stochastic and tunes its weight automatically. PPO (Schulman et al., 2017) is the standard on-policy alternative and is used only in the 2-DOF study.

**Hindsight Experience Replay.** HER (Andrychowicz et al., 2017) relabels failed episodes with goals the agent actually reached, turning a sparse success signal into a dense one. It assumes the reward can be recomputed from (achieved goal, desired goal) alone; Chapter 8 shows what breaks when it can't.

**Residual policy learning.** Silver et al. (2018) and Johannink et al. (2019) add a learned correction to a fixed controller, so learning starts from the controller's competence instead of from scratch. Both report gains where the base controller is good but imperfect. Neither studies a base that tracks a planned path, where the tracker itself may undo the correction (Chapter 9).

**Learned motion planning.** MPNet (Qureshi et al., 2019) and Motion Policy Networks (Fishman et al., 2022) imitate classical planners from large datasets. MPiNets trains on 3 million planning problems from point clouds. DRL planners for industrial arms (e.g. arXiv:2301.05980) and URPlanner (2025) learn collision avoidance directly from distance-based observations. cuRobo (Sundaralingam et al., 2023) is a GPU-parallel classical planner that sets the bar for planning speed.

**Rotation representation and evaluation.** Goals use the continuous 6D rotation representation of Zhou et al. (2019). Results are reported with bootstrap intervals and interquartile means following Agarwal et al. (2021).

## 3 Problem and environment

The task is to drive the gripper's tool point to a target 6D pose (position and orientation) without collisions, using joint-position commands at 25 Hz. Everything runs in MuJoCo with the UR5e model from MuJoCo Menagerie.

**Robot and gripper.** A Robotiq 2F-85 is welded to the flange as one rigid 1.05 kg body, held fully open, with two capsules as its collision geometry. The tool point sits midway between the finger pads, 14.16 cm from the flange. Gravity compensation is on, matching the real UR controller; without it the shoulder sagged 15–32 mrad (2–3 cm at the tool). Physics runs at 0.002 s with a frame skip of 20.

**Action rule.** The policy outputs a ∈ \[−1, 1\]⁶. The joint target integrates it: q\_target ← clip(q\_target + a·Δq\_max), with Δq\_max = 0.04 rad per step and the target held within 0.25 rad of the measured joints. This gives about 0.98 rad/s per joint. Re-basing each step on the measured angle instead capped the arm at 0.2 rad/s, because of the servo's damping-to-stiffness ratio of 0.2 s.

**Observation.** Joint angles as sin/cos plus q/q\_limit, joint velocities, tool position and 6D rotation, the previous action, and q\_target − q. Obstacle stages add four obstacle slots (center, radius, mask) and the clearance of each of the six links and the gripper, clipped at 0.3 m. Goal errors are computed inside the policy from whichever goal is in use (Chapter 8).

**Goals and success.** A goal is a 9-D vector: position plus the first two columns of the rotation matrix. Goals are sampled in joint space and mapped through forward kinematics, rejecting configurations in collision, so every goal is reachable. Success means the tool is within 1 cm and 5° of the goal (2 cm and position only in S1).

| Stage | Task | Obstacles | Episode length | Used in |
| --- | --- | --- | --- | --- |
| S1 | Reach a position | none | 150 steps (6 s) | E1 |
| S2 | Reach a pose | none | 175 steps (7 s) | E2 |
| S3 | Reach a pose | 1–3 fixed spheres | — | debugging only |
| S4 | Reach a pose | 1–4 random spheres per episode | 407 steps (16.3 s) | E5, E5b |

The S4 episode length is 1.25 × the 99th-percentile duration of full RRT-Connect paths on the test set. Collisions end the episode in S3 and S4 only; in S1 and S2, contact with the table or the arm itself is counted but not penalized.

## 4 Method

The hybrid controller is a cheap planner, a tracking controller that follows its path, and a SAC policy that corrects the tracker's command. Two residual designs were tested: an additive correction (E5) and an accumulated offset (E5b).

**Cheap planner (the base).** RRT-Connect in joint space with a budget of N collision checks, N ∈ {1500, 3000, 6000}. The planner sees obstacles inflated by 5 cm (not within 0.3 rad of the start and goal) and runs fewer shortcut iterations than the full planner. When it fails to plan, the base falls back to a straight joint-space line to the nearest collision-free IK solution, ignoring obstacles. During training the budget is drawn per episode from the three values.

**Tracking controller.** The base command feeds the next reference point forward from the current joint target:

```latex
a_{\text{base}} = \operatorname{clip}\!\left(\frac{q_{\text{ref}}(t+1) - q_{\text{target}}}{\Delta q_{\max}},\, -1,\, 1\right)
```

**E5: additive residual.** The policy's output is scaled and added to the base command:

```latex
a = \operatorname{clip}\big(a_{\text{base}} + \alpha\, \pi_{\text{res}}(s),\, -1,\, 1\big)
```

α ramps from 0 to 0.3 over the first 150k steps. The policy starts at the base: the actor's mean head is initialized near zero, its log-std bias at −3, the entropy coefficient at 0.01, and the target entropy at −12. Only the critic trains for the first 50k gradient steps. A penalty w\_res·‖π\_res‖² with w\_res = 0.1 discourages large corrections. The residual sees the reference q\_ref(t) and q\_ref(t+5) and trains without HER, because a relabeled goal would not match the reference planned for the original goal. It was trained with both the dense and the sparse reward.

**E5b: accumulated offset.** The correction builds up into an offset that persists, so it can steer away from the reference:

```latex
o_t = \operatorname{clip}\big(o_{t-1} + \alpha\, \pi_{\text{res}}(s)\, \Delta q_{\max},\, -0.3,\, 0.3\big), \qquad q_{\text{target}} = q_{\text{ref}} + f_t\, o_t
```

The offset changes by at most 0.012 rad per step and can reach 0.3 rad in about 25 steps. The fade f\_t falls linearly to 0 within 5 cm of the goal or over the last 10% of the reference, whichever is sooner, so the target always returns to the true goal. The penalty applies to the offset itself, w\_res·‖o/0.3‖², with sparse reward and otherwise E5's settings.

**Rewards.** Sparse: 0 on success, −1 otherwise. Dense: weighted position and orientation error plus effort and smoothness terms and a success bonus. In obstacle stages both add a clearance-margin penalty below 5 cm. A collision ends the episode with an absorbing penalty, so crashing is always worse than timing out:

```latex
R_{\text{coll}} = \frac{r_{\min}}{1-\gamma} - w_c
```

r\_min is the most negative per-step reward the environment can give, which comes to −207.5 (sparse) and −663.8 (dense) on the UR5e.

**HER reward decomposition.** For the goal-conditioned pure-RL agents, goal-independent terms (collision, clearance, smoothness) are stored per transition as a float and added back whenever HER recomputes the reward for a relabeled goal (Chapter 8).

## 5 Experimental protocol

All agents use Stable-Baselines3 with the same settings, so differences come from the method, not from tuning. Every setting below was fixed in SPEC.md before the runs it applies to.

| Experiment | Stage | Compares | Seeds × steps |
| --- | --- | --- | --- |
| E1 | S1 | TD3+HER vs SAC+HER | 3 × 500k |
| E2 | S2 | SAC sparse+HER vs SAC dense | 2 × 500k |
| E5 | S4 | Pure SAC+HER; residual SAC, dense and sparse reward | 3 × 1M each |
| E5b | S4 | Accumulated-offset residual SAC | 3 × 1M |

**Shared settings.** 3×256 networks, batch 256, 4 parallel environments, 0.25 gradient steps per transition. γ = 0.98 for runs with HER and 0.99 without (E2 dense and all residual runs). Replay buffers hold 500k transitions in E1 and E2 and 1M in E5 and E5b; learning rates are 3e-4 for SAC and 1e-3 for TD3. HER uses "future" relabeling with k = 4. E1 and E2 use a goal-distance curriculum: the maximum joint-space goal distance starts at 0.6 rad and grows by 0.3 rad (up to 3.2) each time success on that level passes 80%, with 20% of goals always drawn from the full range.

**Classical baselines.** B1: analytic IK plus a quintic joint trajectory at the speed limit. B2: full RRT-Connect with a 2 s cap, collision checks every 0.01 rad, 200 shortcut iterations and quintic time-scaling. Base@N: the degraded planner and tracker alone at N checks.

**Test sets.** Frozen before training with disjoint seed ranges: 200 S4 test episodes, plus separate 100-episode validation sets per stage for model selection. Test scenes are kept only if full B2 finds a path.

**Checkpoint rule.** For residual runs, the primary checkpoint is the best validation score after both the α ramp and the critic warm-up have finished; the final 1M checkpoint is also reported. This keeps selection from picking an untrained residual, which is identical to the base.

**Metrics and statistics.** Success, collision rate, final position and orientation error, failure split (collision / timeout / pose miss), path length, jerk and clearance. Trained methods report the seed mean with a 95% bootstrap interval over seeds. Because 2–3 seeds give wide seed intervals, success also carries a pooled Wilson interval over all test episodes, which reflects episode-level variation only.

**Compute.** No GPU: an i5-1240P laptop plus free Colab (2-core) and Kaggle (4-core) CPU sessions, about 2–4 hours per million-step run at 80–160 steps/s.

## 6 Results: reaching without obstacles

Both reaching experiments fell far short of their targets within the step budget. SAC beat TD3 (RQ1), and sparse reward with HER beat dense reward on success but not on final error (RQ2).

**E1, position reaching (S1, target ≥ 95% within 2 cm).** On the full goal distribution, SAC+HER reached 20.7% (seeds 19 / 23 / 20%) and TD3+HER 12.7% (11 / 17 / 10%) after 500k steps. These are final validation scores on 100 episodes per seed; E1 had no separate test-set evaluation. "Stuck" (the arm stops short of the goal) was the main failure: 60% of SAC's failures and 49% of TD3's. Learning was slow because goals lie up to 3.2 rad away in joint space, about 80 control steps at full speed. A pilot with γ = 0.95 (an effective horizon of about 20 steps) learned nothing; the same setup on goals within 0.6 rad reached 93% in 40k steps. Raising γ to 0.98 and adding the goal-distance curriculum gave the best of three 120k-step probes (7%, against 0% for the other two); E1 used that configuration.

**E2, pose reaching (S2, target ≥ 90% within 1 cm and 5°).** SAC was chosen from E1.

| Reward | Success | Final position error | Final orientation error | Table/self contact |
| --- | --- | --- | --- | --- |
| Sparse + HER (s0 / s1) | 4.5% (3 / 6) | 35.4 cm | 67.6° | 38% |
| Dense (s0 / s1) | 0% (0 / 0) | 30.2 cm | 54.2° | 10% |

Sparse with HER is ahead on success, but that rests on 9 vs 0 successes in 200 episodes, and dense ends closer to the goal. S2 has no contact penalty, and the sparse agent touched the table or itself in 38% of episodes, which likely costs pose accuracy.

Both shortfalls are reported as is under the project's one-week rule. They bound what pure RL achieved here and are not a claim about what SAC can reach with more steps.

## 7 Results: motion planning in clutter

The answer to RQ3 is no. The planner alone beats every learned hybrid at every budget, and no hybrid comes near full RRT-Connect.

&#91;embedded content: results/tables/e5.csv · S4 test set, primary checkpoints, seed means\]

The gap is not noise. At 6,000 checks the dense hybrid's seed interval is 72.0–77.0%, entirely below the planner alone's episode-level interval of 85.1–93.4%. Final 1M checkpoints are lower still for E5 (61.5–69.7% dense), while E5b's final checkpoints are slightly better than its primary ones (59.3–63.8%). Pure SAC+HER from scratch succeeds in 0% of episodes.

| Collision rate (%) | 1500 | 3000 | 6000 |
| --- | --- | --- | --- |
| Planner alone | 20.5 | 14.0 | 8.5 |
| E5 additive, dense | 21.5 | 15.3 | 10.3 |
| E5b offset | 42.8 | 41.7 | 39.8 |
| Pure SAC+HER (no planner) | 53.8 | 53.8 | 53.8 |

The two designs lose in different ways. E5 adds at most 2 points of collisions: nearly every lost success is a pose miss at the goal. E5b doubles to triples collisions. Full B2 reaches 99.0% success with 0.5% collisions. Chapter 9 explains both failures.

## 8 Methods findings

Three defects silently broke training. Each produced no error and looked like an algorithm that simply wasn't learning. All three share one cause: anything in a transition that depends on the goal, or is lost when the goal is relabeled, must be recomputed or carried with it.

| # | Defect | Symptom | Fix |
| --- | --- | --- | --- |
| 1 | Goal errors (goal − tool) placed inside the observation | HER relabels the goal but not the errors, so most training samples point at the wrong goal. E1 stayed at 0% after 120–160k steps. | Compute goal errors inside the policy from whichever goal is in use; tests check relabeled samples. |
| 2 | SB3's HER buffer drops info dicts by default (copy\_info\_dict=False) | Collision and clearance penalties vanish from relabeled rewards. Storing the dicts instead costs about 338 bytes per transition. | A buffer subclass that stores the goal-independent penalty as one float (8 bytes) and adds it back on every relabel; the stock buffer now raises an error. |
| 3 | A collision ended the episode with a fixed −10 | Staying alive costs about −50 discounted, so a crash at step 9 (about −17) paid better. Pure SAC crashed in 100% of episodes, within about 9 steps. | An absorbing penalty R\_coll = r\_min/(1−γ) − w\_c, tested so crashing is worse than timing out in both reward modes. |

Defect 3 was found only because the first E5 run's pure-RL arm collided in every episode. After the fix, collisions were spread through the episode like a random policy's (median step 80), confirming the exploit was gone.

## 9 Discussion: the authority–safety dilemma

The two residual designs bracket the problem. A correction bounded tightly enough to stay on the planner's safe path cannot fix the planner's failures; one free enough to fix them leaves the safe path. Neither learned the selective behaviour the claim needs: stay at zero in the 80–90% of episodes where the planner already succeeds, and act only where it fails.

**Measured (findings).**

- **E5 has no authority.** The tracker pulls the target back onto the reference every step, so on every unsaturated step (87% of steps) |q\_target − q\_ref| stays at or below α·Δq\_max = 0.012 rad. Nearly its entire loss is pose misses (12.7–14.2% of episodes, against 0–0.5% for the base); collisions rise by at most 2 points. On paired collision-free episodes at 1500 and 3000 checks, the base ends a mean 0.41 cm and 0.9° from the goal and the hybrid 0.72 cm and 1.3° (at 6000: 1.18 vs 1.48 cm).
- **E5b has authority but saturates.** At 1M steps the median offset at the end of an episode is 0.26–0.30 rad, at or near the 0.30 rad clip limit, in failures and successes alike. Across seeds and checkpoints, 87–96% of collisions happen with |o| ≥ 0.15 rad, and collisions rise from 8.5–20.5% (base) to 36–43%.
- **α sweeps agree.** Over 3 seeds, E5's validation success falls from 0.88 at α = 0 to 0.87, 0.85 and 0.76 at α = 0.1, 0.2 and 0.3 (primary checkpoint), with collisions edging up from 0.12 to 0.14; E5b already drops to 0.70–0.78 validation success at α = 0.1, because α sets only how fast the offset moves, not its limit.

**E5b design history.** E5b went through three versions, each change decided on validation evidence before any test-set evaluation. v0 used the wrong units (up to 0.3 rad of offset change per step). v1 faded the offset only within 5 cm of the goal; since the offset itself kept the tool about 24 cm away, 98% of failures never entered the fade. v2, with a time-based fade and a penalty on the offset, was declared final before its results were known.

**Untested hypotheses for why neither learned selectivity.**

1. The signal for when to act is sparse: only the 10–20% of episodes where the planner fails reward any correction at all.
2. Max-entropy noise drifts through the integrator. SAC is rewarded for staying stochastic, and integrating random actions walks the offset to its limit.
3. w\_res = 0.1 is weak against the reward scale, and the collision penalty arrives only after contact, not on approach.

The 2-DOF arm showed the same pattern on validation (residual 0.88 vs base 0.91 in P2), so the effect is not specific to the UR5e (Chapter 11).

## 10 Limitations and future work

The negative result holds for this setup and budget; it does not show that residual correction over planners cannot work.

**Limitations**

- **Budget.** 500k–1M steps per run, 2–3 seeds, 0.25 gradient steps per transition, and CPU-only compute. More steps or updates per transition could change E1, E2 and the residual results.
- **Simulation only.** Obstacles are spheres with known positions; there is no perception and no real-robot test.
- **One base design.** A single degraded RRT-Connect with a straight-line fallback; a different base or fallback could leave the residual more room.
- **Untuned rewards.** Dense-reward weights and w\_res were set once, not swept.
- **Static scenes.** Moving obstacles (S5) were deferred.

**Future work**

- **Gated residual.** Act only when the planner fails to plan, leaving the 80–90% of successful episodes untouched.
- **Safety-filtered residual.** Project the correction onto actions that keep clearance, or train with a Lagrangian constraint on collisions.
- **TD3 for integrated actions.** A deterministic actor tests hypothesis 2 (entropy-driven drift) directly.
- **Demonstrations.** Pre-train the residual on corrections from the full planner.
- **Real arm.** URSim first, then a UR5e with a depth camera feeding the planner a perceived obstacle map.

## 11 Methodology validation on a planar 2-DOF arm

To be written when the 2-DOF evaluation finishes (PE3, PE4, PE4b, PE8 on exact and ±20% plants, PE9-lite). Settled so far:

- **PE1 (reach a point):** SAC+HER 79%, TD3+HER 64%, DDPG+HER 42%, PPO 0% at 50k steps. SAC first and TD3 second, the same order as on the UR5e.
- **PE2 (reward and curriculum):** sparse+HER with curriculum 79%, without 71%; dense 2–5%.
- **Transfer (PE9-lite, so far):** the two arms rank 3 of 3 compared design choices the same way.

- [ ] Fill PE3/PE4: residual vs base vs pure RL in clutter, gap to best-known path
- [ ] Fill PE4b: accumulated-offset residual on the planar arm
- [ ] Fill PE8: torque control on exact and ±20% mass plants, residual over PD + gravity compensation
- [ ] Final PE9 transfer verdict

## References

- Agarwal, R. et al. (2021). Deep RL at the edge of the statistical precipice. arXiv:2108.13264.
- Andrychowicz, M. et al. (2017). Hindsight Experience Replay. arXiv:1707.01495.
- Fishman, A. et al. (2022). Motion Policy Networks. CoRL 2022, arXiv:2210.12209.
- Fujimoto, S., van Hoof, H., Meger, D. (2018). Addressing function approximation error in actor-critic methods (TD3). arXiv:1802.09477.
- Haarnoja, T. et al. (2018). Soft Actor-Critic. arXiv:1801.01290.
- Johannink, T. et al. (2019). Residual reinforcement learning for robot control. ICRA 2019, arXiv:1812.03201.
- Lillicrap, T. et al. (2015). Continuous control with deep reinforcement learning (DDPG). arXiv:1509.02971.
- Qureshi, A. H. et al. (2019). Motion Planning Networks. arXiv:1806.05767.
- Schulman, J. et al. (2017). Proximal Policy Optimization algorithms. arXiv:1707.06347.
- Silver, T. et al. (2018). Residual Policy Learning. arXiv:1812.06298.
- Sundaralingam, B. et al. (2023). cuRobo: parallelized collision-free robot motion generation. arXiv:2310.17274.
- Zakka, K. et al. (2025). MuJoCo Playground. arXiv:2502.08844.
- Zhou, Y. et al. (2019). On the continuity of rotation representations in neural networks. CVPR 2019, arXiv:1812.07035.
- URPlanner: a universal paradigm for collision-free robotic motion planning based on DRL (2025). arXiv:2505.20175.
- Deep-reinforcement-learning-based path planning for industrial robots using distance sensors as observation (2023). arXiv:2301.05980.
- Kuffner, J., LaValle, S. (2000). RRT-Connect: an efficient approach to single-query path planning. ICRA 2000.
