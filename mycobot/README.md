# MyCobot 280 lane — planner-guided residual on a real arm

Deployment of the project's method (cheap RRT-Connect planner + additive SAC residual) on an Elephant Robotics
MyCobot 280 (M5, suction head). Spec and all numbers: [SPEC.md](SPEC.md) (v0.14).

## What was done
1. **Hardware facts** (pymycobot 4.0.7, COM13): 20 Hz command cycle, joint limits, ~0.5–0.75° servo deadband, ~1° lag.
2. **Simulation:** MuJoCo model from the official URDF (FK within 3.6 mm RMS of the arm), batched IK, same planners,
   same observation layout as the UR5e env, 200-episode test set (horizon 900), challenge set (gaps, clutter, limits,
   far goals) and shape set (boxes and cylinders).
3. **Zero-shot probe:** the UR5e-trained residual on the MyCobot sim matches the cheap planner.
4. **Retraining:** 1M steps × 2 seeds on the MyCobot sim; the residual again equals the planner (88–89.5 / 91.5–92.5 / 93–93.5 %
   success at 1500 / 3000 / 6000 checks, planner 89.5 / 92.5 / 93.5).
5. **Real arm, streaming** (virtual obstacles, first 10 test episodes × B2 / cheap planner / hybrid): 30/30 success, no
   collisions, all final errors < 1 cm and < 4.3° once the final-approach correction (`TrimArm`) is on; without it one B2
   episode stopped 1.46 cm short (J2 under load). Streaming caps the speed at 10°/s and is not smooth.
6. **Real arm, firmware segments** (SPEC M7g): the episode is played in the model, thinned to a few collision-checked
   waypoints, and each is one firmware-planned `send_angles`. Same 30 runs: 30/30 success, no collisions, final errors
   < 0.85 cm and < 3.9°, median wall time 5–6 s against 8–10 s, slowest run 12 s against 45 s.
7. **Videos:** real camera, real joints replayed in the model, and simulation side by side, with the official M5 meshes
   drawn over the model (visual only; `scripts/render_mesh_compare.py`, `assets/m5_meshes/`).

## Honest limits
- The residual does not beat its planner, in simulation or on hardware (n = 10 cannot show small differences).
- Obstacles are given as virtual spheres; nothing is sensed (no camera). Real-object detection is future work.
- The 1 cm / 5° margin is thin (deadband, lag, 0.088° encoder step).
- Motion quality: streaming vibrates slightly and is capped at 10°/s. Segment mode is faster but its smoothness was not
  measured, and in that mode the residual is replayed from the simulation, not corrected from the arm's measurements.
- Hardware runs: one run per method and episode, streaming first and segments later (not randomised). Challenge and
  shape episodes were not run on the arm.

## Commands (from this folder, main venv `../.venv`)
```
python scripts/play.py --testset s4_test --episode 0 --method hybrid --run results/runs/m5/residual_b99f130c_s1 --checkpoint final

# real arm, streaming (COM13, needs pymycobot)
python scripts/run_real.py --episode 0 --method b2 --dry-run          # simulated arm
python scripts/run_real.py --episode 0 --method b2 --execute          # real arm
bash scripts/real_batch.sh                                            # the 30-run streaming batch

# real arm, firmware segments (faster)
python scripts/run_segments.py --episode 0 --method b2 --dry-run      # simulated arm, checks the plan
python scripts/run_segments.py --episode 0 --method b2 --execute --speed 40 --tol 6
python scripts/run_segments.py --probe --execute                      # small multi-joint straight-line test
bash scripts/seg_batch.sh                                             # the 30-run segment batch
python scripts/seg_summary.py                                         # segments vs streaming table

# motion diagnostic (J1 only) and videos
python scripts/motion_diag.py COM13 --only incr,p2p --speeds 20,30,40
python scripts/film_batch.py run  --name seg1 --episodes 0 3 7 --methods b2 hybrid   # one continuous recording
python scripts/film_batch.py make --name seg1                                         # offline: mesh 3-panel videos

python -m pytest tests/test_mcrl.py -q
```
Close the Windows Camera app before filming. The arm moves in every `--execute` command; keep the workspace clear.
