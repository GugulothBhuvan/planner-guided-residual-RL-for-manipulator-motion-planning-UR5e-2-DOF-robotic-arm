# ur5e-rl-planner

RL trajectory & motion planning for a UR5e with a rigid Robotiq 2F-85 in MuJoCo. The design is in [SPEC.md](SPEC.md).

## Setup

```bash
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
.venv/Scripts/python -m pip install -r requirements.lock
.venv/Scripts/python -m pip install -e ".[dev]"
```

The menagerie UR5e and 2F-85 models are vendored in `assets/menagerie/` (commit in `assets/menagerie/COMMIT`).

## Day-1 commands

```bash
python scripts/build_scene.py        # regenerate assets/scene_ur5e.xml (+ .meta.json)
python -m pytest                     # FK, rotations, clearance, env API
python scripts/profile_step.py       # physics / clearance / reset timings + actuator check
python scripts/bench_throughput.py --mode dense --threads 1   # SPEC §11 throughput gate
```

Run `bench_throughput.py` on every machine type (laptop, Colab, Kaggle) before launching training.
