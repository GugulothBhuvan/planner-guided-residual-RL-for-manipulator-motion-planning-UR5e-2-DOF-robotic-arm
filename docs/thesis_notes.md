# Thesis notes

Running notes for the write-up: findings, failures and the reasoning behind design decisions. Dated; newest last.

## 2026-10-26 — HER and anything that depends on the goal

**Principle.** HER rewrites the goal of stored transitions. Every quantity in a transition is therefore either
goal-dependent, and must be recomputed from the relabeled goal, or goal-independent, and must be carried over
unchanged. Getting either half wrong corrupts the relabeled samples silently: training runs, losses look normal,
and nothing learns.

Two instances in this project, both found by checking what actually reaches the learner:

1. **Rewards — safety terms dropped.** Collision, clearance-margin and smoothness penalties do not depend on the
   goal, but SB3's `HerReplayBuffer` passes empty info dicts to `compute_reward` by default, so they vanish on
   relabeled samples. Fix: `SafetyHerReplayBuffer` stores the goal-independent term (8 B per transition) and adds it
   back. Our `compute_reward` raises on missing terms instead of defaulting to zero. (SPEC §3.1 part 2, §6.7.)
2. **Observations — stale goal features.** The v0.4 observation contained the position and rotation error to the
   goal. HER relabeled `desired_goal` but those features still described the original goal, so on ~80% of samples
   (HER k = 4) the critic saw one goal in the observation and was rewarded for another. E1's first launch reached
   0% validation success at 120k–160k steps. Fix: `observation` is goal-independent and `GoalErrorExtractor`
   computes [desired − achieved] inside the policy, so it always matches the sample's goal. (SPEC §6.4.)

Both are now guarded by tests: relabeled rewards equal `compute_reward` with the original safety terms, bit for bit;
`observation` is unchanged when the goal changes.

3. **Termination — crashing paid (found 2026-10-28).** Not a HER bug, but the same kind of silent failure: the
   reward was correct term by term and wrong as a whole. A collision ended the episode with a one-off −w_c = −10,
   while staying alive cost −1 per step (sparse; ≈ −50 discounted at γ 0.98, up to −407 undiscounted over the S4
   horizon). Ending the episode by crashing was therefore the best policy whenever success was unlikely. Pure
   SAC+HER on S4 learned exactly that in all three E5 seeds: 0% success, 100% collisions, every validation episode
   over after ~9 steps (PE3 on the planar arm: ~15 steps). The residual agents drifted the same way more mildly
   (collision rate 0.12 → 0.16–0.18). Termination turns "the episode ends" into a reward the designer did not write.
   **Fix (SPEC §6.7):** the collision step pays the value of staying in the worst state forever,
   R_coll = r_min/(1−γ) − w_c, with r_min the analytic per-step minimum of that reward mode; episodes still end on
   collision. A penalty that depends on the remaining steps (T − t) was rejected: t is not in the observation, so
   that reward would be non-Markov. Tested: r_min bounds every observed per-step reward, the env pays exactly
   R_coll on a collision step, and R_coll is below the worst-case return of surviving any number of further steps,
   in both reward modes and on both arms. The v1 runs are archived as `e5_v1_crash_exploit` and
   `pe3_v1_crash_exploit`.

## 2026-10-26 — Discount factor vs task horizon with HER

E1's second launch used γ = 0.95, the common HER default. Its effective horizon 1/(1−γ) ≈ 20 control steps is far
shorter than the ~80 steps a 3.2 rad goal needs at 1 rad/s and 25 Hz, so for distant goals the sparse-reward value
is nearly flat and gives the actor no direction. Symptom: 0–6% success at 200k–250k steps while the same pipeline
learned near goals (≤ 0.6 rad, 5 cm) to 93% in 40k steps. Lesson: choose γ from the task's step horizon, not from
FetchReach defaults (50-step episodes, goals a few cm away). The S1 probe (γ 0.98 / 0.99, ± goal-distance curriculum)
picks the replacement (SPEC §4).

## 2026-10-26 — "Optimal" reference paths on a grid (2-DOF)

Grid A* on an 8-connected C-space grid is optimal only for the grid metric (axis and diagonal moves), which
overestimates the true joint-space length by up to ~8%. On the first test query a raw RRT-Connect path was shorter
than raw A* (2.57 vs 2.69 rad). Using raw A* as the reference would let RL trajectories look "better than optimal"
and bias the optimality gap (P-RQ2). Reference = A* + any-angle shortcut smoothing; the spec's "A* ≤ RRT" test was
corrected accordingly.

## 2026-10-26 — Simulation modeling assumptions found on Day 1

- **Integrated joint targets.** Re-anchoring position targets on the measured q under the menagerie servo (kv/kp =
  0.2 s) capped joint speed at ~0.2 rad/s instead of 1 rad/s; ~0% of goals were reachable in the horizon.
- **Gravity compensation.** Without it the position servos sag 15–32 mrad (2–3 cm at the TCP) below every target;
  the real UR controller compensates internally. Modeled with MuJoCo actuator gravity compensation; the 2-DOF torque
  model deliberately has none.

## 2026-10-28 — Dense reward lost to sparse + HER on both arms (PE9 transfer result)

The sparse-vs-dense design choice ranked the same way on the planar arm and on the UR5e:

- **2-DOF, PE2 (P1, SAC, 10 seeds × 50k steps, success at 1 cm):** sparse + HER 0.79 with the goal curriculum (PE1's
  run) and 0.71 without; dense 0.05 with the curriculum and 0.02 without.
- **UR5e, E2 (S2, SAC, 2 seeds × 500k steps, success at 1 cm / 5°):** sparse + HER 4.5% mean (s0 3%, s1 6%), dense
  0% (0%, 0%). Both far below the S2 target (shortfall). Final position error favours *dense*: 30.2 cm mean (31.3 /
  29.1) vs 35.4 cm (31.1 / 39.7); orientation error 54.2° vs 67.6°. Sparse + HER also touches the table or itself far
  more (30% / 47% of validation episodes vs 12% / 9%); S2 has no contact penalty, so contact costs nothing.

Same ordering on both arms, so the cheap planar experiment predicted the expensive one for this choice: a PE9 data
point for "design choices transfer from 2-DOF to 6-DOF". Caveats for the write-up: E2 has two seeds and both UR5e success
rates are near zero (9 of 200 validation episodes vs 0 of 200), so the UR5e side shows the direction, not the size,
of the gap, and only on success: on final distance the ordering reverses, so the transfer claim must be stated for
success at the threshold, not for "gets closer"; the dense reward was not tuned
(weights from §6.7), and HER is the confound: "sparse + HER vs dense without HER" compares reward shape and goal
relabeling together, as specified for E2/PE2.

## 2026-10-28 — The learned residual makes a good base worse (E5 α sweep, PE8 nominal plant)

- **UR5e, E5 (S4):** residual SAC s0's 450k checkpoint, replayed on the 100 S4 validation episodes with the residual
  scale α fixed (same episodes, seeds and base-budget draws as training validation; deterministic actions). Success /
  collision: α = 0 (base alone) 0.88 / 0.12, α = 0.1 0.88 / 0.13, α = 0.2 0.82 / 0.17, α = 0.3 (training value)
  0.66 / 0.17. The residual never improves on the degraded base at any α and degrades it monotonically as α grows. The
  training curve says the same: validation sits at the base line (0.88) while the actor is frozen (critic warm-up, to
  ~202k steps), then drops to 0.63–0.74 once the actor trains (0.71 at 500k).
- **2-DOF, PE8 (P5a torque, nominal plant):** B5' (PD + gravity on the nominal model) alone reaches 100% of the 100
  validation episodes (0.45 cm); residual-on-B5' s0 reaches 57% at 300k, pure SAC+HER s0 0%.
- **Reading:** a near-perfect base leaves the residual nothing to fix, so anything it learns can only add error. On
  the nominal plant B5' is exact by construction, so PE8's case for the residual rests entirely on the ±20% mass
  plants, where B5''s gravity term is wrong. For E5 the degraded base still fails 12% of validation episodes, so there
  is room; the residual does not find it in 500k steps. For the write-up: report the α sweep next to the budget figure
  (it separates "the residual learned nothing useful" from "the residual learned something harmful"), and select
  residual checkpoints only after the actor has started training, or the "hybrid" result is the base in disguise
  (SPEC §9.2 checkpoint rule).

## 2026-10-29 — Residual warm-up must scale with run length (PE3/PE4 near-miss)

- The residual settings were tuned for 1M-step E5 runs: a 50k-gradient-step critic warm-up at 0.25 gradient steps per
  transition freezes the actor for the first ~202k env steps. Copied unchanged to the 150k-step planar runs (PE3/PE4),
  the actor would never have trained, and the "hybrid" would have been the base alone, reported as a learned result.
  Caught before launch; PE3/PE4 now keep E5's fractions (α ramp over 15% of the run, actor trains from 20%).
- For the write-up: state warm-up and ramp as fractions of the training budget, not absolute step counts, and check
  for every residual run that the evaluated checkpoint comes after the actor started training (the §9.2 rule does
  this). PE8 (300k) kept the absolute settings because two of its runs had already finished: its actor trains only for
  the last third of the run, which is a limitation to state.

## 2026-10-29 — E5 validation curves, all seeds (before the test-set evaluation)

- Dense residual, validation success at 200k / 500k / 1M: s0 0.88 / 0.71 / 0.71, s1 0.88 / 0.72 / 0.72,
  s2 0.88 / 0.74 / 0.68. Sparse residual: s1 0.88 / 0.54 / 0.60, s2 0.88 / 0.71 / 0.54 (s0 still training). Pure
  SAC+HER at 1M: 0% success on every seed, collision 37% / 66% / 56% (s0 / s1 / s2).
- Every residual seed, dense or sparse, sits exactly at the base line (0.88 / 0.12) while the actor is frozen and falls
  below it once the actor trains, and none recovers by 1M. The sparse reward does not change this (it is lower, not
  higher). This is the E5 α-sweep finding on all seeds: on S4 the learned residual degrades the cheap planner instead
  of repairing it. The test-set evaluation (primary and final checkpoints) decides whether this holds on held-out
  episodes and at each budget.
- Pure SAC+HER without a planner does not solve S4 at all in 1M steps (0% on validation, every seed).

## 2026-10-29 — Torque baselines on perturbed plants (PE8 set-up check)

- On 50 test episodes, B5 (computed torque) and B5' (PD + gravity) reach 100% on the exact plant and 10–30% on the
  ±20% mass plants. Both still end close (~2 cm) but outside the 1 cm and 0.1 rad/s condition, because the nominal
  gravity term is wrong. So PE8's perturbed plants pose exactly the kind of small, systematic error a residual is
  meant to remove, in contrast with the nominal plant, where B5' leaves nothing to fix.

## 2026-10-29 — E5 headline: the residual does not repair the cheap planner (S4 test set)

- 200 S4 test episodes per seed; dense residual 3 seeds, sparse residual 2 seeds (s0 pending). Success, base-only vs
  dense hybrid at the primary / final checkpoint: 1500 checks 79.5 vs 66.2 / 61.5%; 3000 checks 86.0 vs 71.5 / 66.5%;
  6000 checks 90.0 vs 74.8 / 69.7%. Sparse hybrid (primary): 69.2 / 74.2 / 77.5% on 2 seeds; with s0, 3 seeds:
  66.3 / 70.8 / 74.0% (corrected 2026-10-30, results/tables/e5.md). Collision rates are the base's at
  every budget (≈ 21 / 15 / 10%). Full B2 99.0% success, 0.5% collision; pure SAC+HER 0% success, 54% collision.
- The 95% CI over seeds of the hybrid is entirely below base-only at every budget: this is a clear negative result for
  contribution 1 as stated (§3.1). The residual does not trade collisions for success; collisions stay where the base
  puts them, and the residual only costs pose accuracy. The base's failures are collisions on its straight-line
  fallback, and a residual of at most 0.3 of the action range, trained on a dense pose reward, does not learn to steer
  around them.
- Checkpoint selection exposes the trend: the best-validation checkpoint after the actor starts training is 600k–800k
  for dense seeds and 300k (the first eligible) for both sparse seeds; the final 1M checkpoint is worse than the primary
  in every arm and budget. More residual training makes the hybrid worse.
- For the write-up: report it as a negative result with its mechanism (α sweep: monotone decline from the base at
  α = 0; collision unchanged), not as a tuning failure. What remains is whether the planar arm shows the same
  (PE3/PE4), and whether the residual helps where the base has a systematic error rather than a geometric one (PE8's
  ±20% plants).

## 2026-10-29 — Why the E5 residual fails: the base cancels it every step (diagnostic)

- E5 is reported as a negative result, unchanged. Mechanism, measured on the S4 test episodes: the base tracker sets
  its action from `q_ref(t+1) - q_target` every step, so whatever the residual added to `q_target` last step is
  taken back this step. On every unsaturated step (87%) the residual's displacement of the target is at most
  α·Δq_max = 0.012 rad (measured max 0.0120); larger excursions happen only while the base is saturated pulling
  back. 0.012 rad is far too little to steer around an obstacle, but enough to spoil a 1 cm / 5° final pose.
- The outcome split shows exactly that: the hybrid's collisions equal the base's at every budget, and all of its
  extra failures are pose misses (12.7–15.8% of episodes vs 0–0.5% for the base; 3 seeds per arm, corrected
  2026-10-30 from the 2-seed sparse value 10–14%). On episodes both finish without contact, the hybrid ends at
  0.72 cm (dense) / 0.75 cm (sparse) and 1.3° vs the base's 0.41 cm / 0.9° (means at 1500 and 3000 checks).
- Lesson for the write-up: in a residual-on-tracker design, the tracker's feedback defines the residual's authority.
  An additive residual on a re-aiming position tracker is a bounded perturbation, not a correction, however long it
  trains. This is a property of the architecture, so it can be shown analytically and confirmed by measurement.

## 2026-10-29 — E5b: pre-registered second iteration (integrated offset)

- Registered in SPEC §7.3 / §8 before any training: the residual integrates a persistent joint offset
  (o ← clip(o + α·π_res, ±0.3 rad), target q_ref + s·o, s fading the offset to 0 within 5 cm of the goal), sparse
  reward, E5's warm-up / ramp / w_res, 3 seeds × 1M, the same S4 test-set evaluation and checkpoint rule.
- It is reported whatever it shows. E5 stays the headline negative result; E5b is the follow-up that tests whether
  giving the residual real authority (a lasting offset) lets it remove the base's collisions.

## 2026-10-29 — E5b offset step, and PE4b

- E5b's offset step is α·π_res·Δq_max (≤ 0.012 rad per step; ±0.3 rad in ~25 steps), as intended; a first launch
  that read α·π_res in radians was stopped at 100k and archived (`e5b_v0_rad_units`), and E5b restarted from 0. The
  pre-registration (SPEC §7.3) records both.
- PE4b, recorded before training: the same accumulated residual on the planar P3 stage (5 seeds × 150k). Together with
  E5b it asks, on both arms, whether giving the residual lasting authority changes the outcome. PE9-lite reports
  whether the answer agrees across arms.

## 2026-10-29 — E5b iterations, all before any E5b test-set evaluation

Each change was made on validation evidence only. The S4 test set has not been touched for E5b.

| Version | Change | Evidence that triggered the next version |
|---|---|---|
| v0 | integrated offset, increment α·π_res in radians (up to 0.3 rad per step) | Units did not match the design intent (α·π_res·Δq_max, ≤ 0.012 rad per step); stopped at 100k (validation 0.58), archived `e5b_v0_rad_units` |
| v1 | increment α·π_res·Δq_max; fade within 5 cm of the goal only; w_res on π_res | Validation collapsed once the actor trained (0.88 at 200k → 0.04 at 225k, collision 0.50). Diagnostic at 300k: 0/100 success, 54% collisions; 95% of failures end with fade = 1 and |o| ≥ 0.15 rad (median 0.300, the clip); median |o| at collision 0.299 rad. Same at 700k (0.05). The saturated offset keeps the tool > 5 cm from the goal, so the distance fade never engages. Archived `e5b_v1_dist_fade` |
| v2 | fade also linear over the last 10% of the reference (min of the two); w_res charges ‖o/0.3‖² | Final iteration (user decision): reported whatever it shows, no v3 |

- Reading v1: given lasting authority, the policy pushed the offset to its limit and kept it there. That made it
  collide more than the base did (0.54 vs 0.12), and it never arrived. With the penalty on π_res, holding a large
  offset cost nothing; v2 makes the offset itself pay.
- For the write-up: report all three versions and the evidence for each change, and state that every change was made
  before any E5b test-set evaluation. PE4b on the planar arm uses v2 only; its v1 runs are archived unreported.

## 2026-10-29 — E5b v2 at 300k: the fade is fixed, the offset still saturates (analysis only)

- Validation (s0, 100 episodes): 0.88 / 0.12 while the actor was frozen; once it trained, 0.49 / 0.51 (225k),
  0.54 / 0.46 (250k), 0.52 / 0.48 (275k). Diagnostic at 300k, v1 → v2: success 0.00 → 0.46, collision 0.54 → 0.54.
  Every v2 failure is a collision and every non-colliding episode succeeds: the reference fade fixed the
  end-of-episode problem that v1 had. But the offset still saturates (median |o| at the last step 0.300 rad, the clip,
  in failures and successes alike; 96% of collisions with |o| ≥ 0.15 rad), and collisions are at v1's level.
- Likely reason, w_res against the reward scale (sparse reward, γ = 0.99, w_res = 0.1): every unsuccessful step
  costs −1, so the discounted cost of never succeeding is −1/(1−γ) = −100. One joint held at the 0.3 rad limit costs
  w_res·(0.3/0.3)² = 0.1 per step, or 10 discounted (all six joints at most 0.6 per step, 60 discounted). Against the
  time cost the offset is cheap, and the penalty does not stop the policy from pushing it to the limit. A collision
  (absorbing, −(1 + 6·w_res)/(1−γ) = −160) costs only 1.6× a failed episode that runs to the horizon, so it is not
  avoided strongly either.
- Analysis only: E5b v2 is final (no v3) and runs unchanged to 1M on all seeds; it is reported as recorded.

## 2026-10-30 — E5b v2 result: lasting authority makes the residual collide more

- S4 test set, 3 seeds × 200 episodes. Success, base-only vs E5b v2 (primary / final checkpoint): 1500 checks 79.5 vs
  57.2 / 59.3%, 3000 86.0 vs 58.3 / 62.0%, 6000 90.0 vs 60.0 / 63.8%. Collision: 20.5 vs 42.8 / 40.7%, 14.0 vs
  41.7 / 38.0%, 8.5 vs 39.8 / 36.2%. E5b v2 is below base-only, and below the E5 dense and sparse hybrids, at every
  budget, with collisions two to four times the base's.
- The two iterations bracket the problem. E5's additive residual is cancelled every step (≤ 0.012 rad of authority):
  it cannot detour, so it changes nothing about collisions and only costs accuracy. E5b's integrated offset has the
  authority to detour (±0.3 rad, persistent), and the policy uses it, but it drives the offset to its limit and
  into obstacles (88–96% of collisions happen with |o| ≥ 0.15 rad). On this budget (1M steps, sparse reward) the
  residual never learns a selective, obstacle-aware correction in either form.
- With the fade fixed, more training helps E5b slightly (final > primary everywhere, collisions 0.54 → 0.29–0.40 on
  validation from 300k to 1M), unlike E5, where more training made the hybrid worse. The trend is still far from the
  base line, and E5b v2 is final (no v3), so it is reported as is.
- α sweep: at α = 0.1 validation success already drops from 0.88 to 0.70–0.78 and collisions double. α scales the
  rate at which the offset changes, not its limit, so any α > 0 lets the policy reach the same saturated offset.

## 2026-10-30 — E5 + E5b: the authority–safety dilemma (UR5e experiments frozen)

UR5e experiments are frozen; there is no further E5 work. E5 and E5b are reported together as two sides of one
dilemma for residual RL on top of a tracking planner.

**Findings (measured, S4 test set and validation diagnostics):**
- *Bounded residual = no authority.* E5's additive residual is re-cancelled by the base tracker every step; on
  unsaturated steps it moves the target at most α·Δq_max = 0.012 rad (measured max 0.0120). It cannot detour: hybrid
  collisions equal the base's at every budget, and its whole loss is pose misses (12.7–15.8% of episodes). Hybrid success
  is below base-only at every budget (66–75% vs 80–90%), dense or sparse reward alike.
- *Integrated residual = leaves the safe envelope.* E5b's persistent offset (±0.3 rad) has the authority to leave the
  reference, and the policy uses it: the offset saturates (median |o| 0.30 rad at episode end, in failures and
  successes) and 88–96% of collisions happen with |o| ≥ 0.15 rad. Collisions rise to 36–43% (base 8.5–20.5%), and
  success is below base-only and below E5 at every budget (57–64%).
- Neither form learns a selective, obstacle-aware correction within 1M steps. The residual either cannot leave the
  reference or does not stay near it.
- Process: E5b went through v0 (wrong units), v1 (distance-only fade) and v2 (reference fade, offset penalty), all
  changed on validation evidence before any test-set evaluation. v2 is final.

**Hypotheses (not tested; for discussion):**
- *Sparse signal for when to act.* The reward says nothing about which steps need a correction: the base succeeds on
  most episodes, so the policy gets no local signal to separate the few steps where a detour helps from the many
  where it harms.
- *Max-entropy plus integrator drift.* SAC's entropy bonus keeps the action noisy; integrated over hundreds of steps,
  that noise becomes a drifting offset that the clip turns into a saturated one.
- *w_res too weak relative to the reward scale.* At γ = 0.99 with −1 per step, holding a joint at the offset limit
  costs ~10 discounted against ~100 for an unsuccessful episode, and a collision costs only 1.6× a timeout, so
  neither the offset nor collisions are expensive enough.

**Future work:**
- *Gated residual:* a learned or rule-based gate that lets the residual act only near predicted conflicts (e.g. low
  clearance on the base path), so the default is the base.
- *Safety-filtered residual:* project the residual's target onto a collision-free set (clearance check or control
  barrier function) before execution, so authority never costs safety.
- *TD3 for integrated actions:* a deterministic policy with explicit, decaying exploration noise, to avoid
  max-entropy drift when actions are integrated.
