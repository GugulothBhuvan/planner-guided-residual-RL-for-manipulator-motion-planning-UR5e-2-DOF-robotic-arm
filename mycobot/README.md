# MyCobot 280 lane — planner-guided residual on a real arm

Deployment of the project's method (cheap RRT-Connect planner + additive SAC residual) on an Elephant Robotics
MyCobot 280 (M5, suction head). Spec and all numbers: [SPEC.md](SPEC.md) (v0.4).

## What was done
1. **Hardware facts** (pymycobot 4.0.7, COM13): 20 Hz command cycle, joint limits, ~0.5–0.75° servo deadband, ~1° lag.
2. **Simulation:** MuJoCo model from the official URDF (FK within 3.6 mm RMS of the arm), batched IK, same planners,
   same observation layout as the UR5e env, 200-episode test set (horizon 900).
3. **Zero-shot probe:** the UR5e-trained residual on the MyCobot sim matches the cheap planner.
4. **Retraining:** 1M steps × 2 seeds on the MyCobot sim; the residual again equals the planner (88–89.5 / 91.5–92.5 / 93–93.5 %
   success at 1500 / 3000 / 6000 checks, planner 89.5 / 92.5 / 93.5).
5. **Real arm** (virtual obstacles, first 10 test episodes × B2 / cheap planner / hybrid): 30/30 success, no collisions,
   all final errors < 1 cm and < 4.3° once the final-approach correction (`TrimArm`) is on; without it one B2 episode
   stopped 1.46 cm short (J2 under load).

## Honest limits
- The residual does not beat its planner, in simulation or on hardware (n = 10 cannot show small differences).
- Obstacles are given as virtual spheres; nothing is sensed (no camera). Real-object detection is future work.
- The 1 cm / 5° margin is thin (deadband, lag, 0.088° encoder step). Slight vibration during motion (new command every 50 ms).
- Hardware runs: one run per method and episode; v1 (no correction) and v2 were not randomised in order.

## Commands (from this folder, main venv)
```
python scripts/play.py --testset s4_test --episode 0 --method hybrid --run results/runs/m5/residual_b99f130c_s1 --checkpoint final
python scripts/run_real.py --episode 0 --method b2 --dry-run          # simulated arm
python scripts/run_real.py --episode 0 --method b2 --execute          # real arm (COM13, needs pymycobot)
bash scripts/real_batch.sh                                            # the 30-run batch
python -m pytest tests/test_mcrl.py -q
```
