"""One-off port of the UR5e env / planners / scripts to the MyCobot 280 (documents every change from the UR5e code).

Generates mcrl/envs/{pose_env,residual,clearance}.py, mcrl/planners/{baselines,collision}.py and
scripts/{train,evaluate,make_testsets}.py by text substitution of the UR5e sources, asserting each substitution hits.
Re-run is idempotent. See mycobot/SPEC.md for the parameter decisions.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
UR = HERE.parent
SRC = UR / "src" / "ur5rl"


def port(src, dst, subs, header):
    text = Path(src).read_text(encoding="utf-8")
    for old, new in subs:
        assert old in text, f"{Path(src).name}: pattern not found: {old[:70]!r}"
        text = text.replace(old, new)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(f'"""{header}"""\n' + text if not text.startswith('"""') else text.replace(
        '"""', f'"""[MyCobot 280 port of {Path(src).name}; see scripts/port_from_ur5.py]\n', 1), encoding="utf-8")
    print("wrote", dst.relative_to(HERE))


# ---------------------------------------------------------------------------------------------- envs
port(SRC / "envs" / "pose_env.py", HERE / "mcrl" / "envs" / "pose_env.py", [
    ("from ur5rl.envs.clearance import ROBOT_BODIES, Clearance\nfrom ur5rl.planners.ik import DH_A, DH_D, fk_flange\n",
     "from mcrl.envs.clearance import ROBOT_BODIES, Clearance\nfrom mcrl import robot\n"),
    ('SCENE_XML = Path(__file__).resolve().parents[3] / "assets" / "scene_ur5e.xml"\n'
     'HOME_Q = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])\n'
     'SHOULDER = np.array([0.0, 0.0, 0.163])  # shoulder-lift axis height (DH d1)',
     'SCENE_XML = Path(__file__).resolve().parents[2] / "assets" / "scene_mc280.xml"\n'
     'HOME_Q = robot.HOME_Q\nSHOULDER = robot.SHOULDER  # J2 axis'),
    ("    dq_max: float = 0.04  # rad / control step\n    target_lead_max: float = 0.25  #",
     "    dq_max: float = robot.DQ_MAX  # rad / control step (hardware-limited, SPEC M3)\n"
     "    target_lead_max: float = robot.TARGET_LEAD_MAX  #"),
    ("    joint_limit: float = np.pi  # Week 1: +-pi on every joint (narrows the model's +-2pi)",
     "    joint_limit: float = np.pi  # the scene's own ranges (guide's GUI limits) are tighter"),
    ("    frame_skip: int = 20\n    timestep: float = 0.002\n    home_noise: float = 0.3\n"
     "    obstacle_radius: tuple[float, float] = (0.04, 0.10)",
     "    frame_skip: int = robot.FRAME_SKIP\n    timestep: float = 0.002\n    home_noise: float = 0.3\n"
     "    obstacle_radius: tuple[float, float] = (0.015, 0.035)"),
    ("    d_safe: float = 0.05\n", "    d_safe: float = 0.02\n"),
    ("class UR5ePoseEnv(gym.Env):", "class MyCobotPoseEnv(gym.Env):"),
    ("""        d0 = mujoco.MjData(m)
        mujoco.mj_forward(m, d0)
        tcp_offset = float(np.linalg.norm(d0.site_xpos[self.tcp_id] - fk_flange(d0.qpos[:6])[:3, 3]))
        self.reach = float(np.abs(DH_A).sum() + DH_D[3:].sum() + tcp_offset)""",
     "        self.reach = robot.REACH"),
    ("            if self.cfg.goal_min_reach is not None and np.linalg.norm(p - SHOULDER) < self.cfg.goal_min_reach:\n"
     "                continue\n", "            if self.cfg.goal_min_reach is not None and np.linalg.norm(p - SHOULDER) < self.cfg.goal_min_reach:\n"
     "                continue\n"),
    ("def _clear_at(self, q, centers, radii, margin=0.02):", "def _clear_at(self, q, centers, radii, margin=0.01):"),
    ("self.np_random.uniform(-0.15, 0.15, 3)", "self.np_random.uniform(-0.05, 0.05, 3)"),
    ("if not (0.2 < radial < 0.85 and r < c[2] < 1.0):", "if not (0.04 < radial < 0.32 and r < c[2] < 0.45):"),
    ("self.q_start = self.q_goal = HOME_Q.copy()", "self.q_start = self.q_goal = HOME_Q.copy()"),
], "Goal-conditioned MyCobot 280 + suction head pose-reaching env (port of the UR5e env, SPEC M3)")

port(SRC / "envs" / "clearance.py", HERE / "mcrl" / "envs" / "clearance.py", [
    ('ROBOT_BODIES = ("shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link",\n'
     '                "wrist_3_link", "gripper")',
     'ROBOT_BODIES = ("link1", "link2", "link3", "link4", "link5", "link6", "tool")'),
], "Analytic robot-obstacle clearance for the MyCobot 280 (port of the UR5e module)")

port(SRC / "envs" / "residual.py", HERE / "mcrl" / "envs" / "residual.py", [
    ("from ur5rl.planners.baselines import ReferenceTracker, goal_configs, plan_b2\n"
     "from ur5rl.planners.collision import CollisionChecker\nfrom ur5rl.planners.quintic import time_scale\n",
     "from mcrl import robot\nfrom mcrl.planners.baselines import ReferenceTracker, goal_configs, plan_b2\n"
     "from mcrl.planners.collision import CollisionChecker\nfrom ur5rl.planners.quintic import time_scale\n"),
    ("dt=base.cfg.frame_skip * base.cfg.timestep)",
     "dt=base.cfg.frame_skip * base.cfg.timestep,\n                                           v_max=robot.V_MAX, a_max=robot.A_MAX)"),
], "Residual env for the MyCobot 280 (port of the UR5e module)")

# ---------------------------------------------------------------------------------------------- planners
port(SRC / "planners" / "collision.py", HERE / "mcrl" / "planners" / "collision.py", [
    ("from ur5rl.envs.pose_env import SCENE_XML", "from mcrl.envs.pose_env import SCENE_XML"),
], "MyCobot 280 configuration collision checker for planners (port of the UR5e module)")

port(SRC / "planners" / "baselines.py", HERE / "mcrl" / "planners" / "baselines.py", [
    ("from ur5rl.planners.collision import CollisionChecker\nfrom ur5rl.planners.ik import ik_tcp\n"
     "from ur5rl.planners.quintic import V_CAP, time_scale\n",
     "from mcrl import robot\nfrom mcrl.planners.collision import CollisionChecker\nfrom mcrl.planners.ik import ik_tcp\n"
     "from ur5rl.planners.quintic import time_scale\n"),
    ("    sols = ik_tcp(goal_pose(env), flange_to_tcp(env))\n",
     "    sols = ik_tcp(goal_pose(env), flange_to_tcp(env), env.q_lo, env.q_hi, env.q_start)\n"),
    ("ref, dur = time_scale(wp, dt=env.cfg.frame_skip * env.cfg.timestep, v_max=V_CAP, a_max=None)",
     "ref, dur = time_scale(wp, dt=env.cfg.frame_skip * env.cfg.timestep, v_max=robot.V_CAP, a_max=None)"),
    ("    ref, dur = time_scale(path, dt=env.cfg.frame_skip * env.cfg.timestep)\n",
     "    ref, dur = time_scale(path, dt=env.cfg.frame_skip * env.cfg.timestep, v_max=robot.V_MAX, a_max=robot.A_MAX)\n"),
], "Classical baselines on the MyCobot 280 (port of the UR5e module)")

# ---------------------------------------------------------------------------------------------- scripts
port(UR / "scripts" / "make_testsets.py", HERE / "scripts" / "make_testsets.py", [
    ("from ur5rl.envs.pose_env import EnvConfig, UR5ePoseEnv\nfrom ur5rl.planners.baselines import plan_b2\n"
     "from ur5rl.planners.collision import CollisionChecker\n",
     "from mcrl.envs.pose_env import EnvConfig, MyCobotPoseEnv as UR5ePoseEnv\nfrom mcrl.planners.baselines import plan_b2\n"
     "from mcrl.planners.collision import CollisionChecker\n"),
    ('    ("s1_test", "s1", 200, 3_000_000, {}),\n    ("s2_test", "s2", 200, 3_100_000, {}),\n', ""),
    ('    ("s4_ood_test", "s4", 100, 3_300_000, dict(n_obstacles=(4, 4), obstacle_radius=(0.08, 0.14), goal_min_reach=0.75)),\n', ""),
    ('    ("s1_val", "s1", 100, 1_000_000, {}),\n    ("s2_val", "s2", 100, 1_100_000, {}),\n', ""),
    ('OUT = ROOT / "testsets"\n', 'OUT = ROOT / "testsets"\nMAX_B2_DURATION = 20.0  # s: episodes B2 cannot finish within 20 s are rejected (SPEC M4)\n'),
    ("            b2.append((p.ok, p.duration, p.plan_time))",
     "            if p.ok and p.duration > MAX_B2_DURATION:\n                rejected += 1\n                seed += 1\n                continue\n"
     "            b2.append((p.ok, p.duration, p.plan_time))"),
    ("rule=\"max(200, ceil(1.25 * p99 B2 duration / 0.04 s))\"", "rule=\"max(200, ceil(1.25 * p99 B2 duration / 0.05 s))\""),
], "Frozen MyCobot test / validation sets (S4 only) and the S4 horizon (port of the UR5e script)")

port(UR / "scripts" / "evaluate.py", HERE / "scripts" / "evaluate.py", [
    ("from ur5rl.envs.pose_env import EnvConfig, UR5ePoseEnv\nfrom ur5rl.envs.residual import ResidualEnv\n",
     "from mcrl.envs.pose_env import EnvConfig, MyCobotPoseEnv as UR5ePoseEnv\nfrom mcrl.envs.residual import ResidualEnv\n"),
    ("from ur5rl.planners.baselines import ReferenceTracker, plan_b1, plan_b2\nfrom ur5rl.planners.collision import CollisionChecker\n",
     "from mcrl.planners.baselines import ReferenceTracker, plan_b1, plan_b2\nfrom mcrl.planners.collision import CollisionChecker\n"),
    ("import checkpoints  # scripts/checkpoints.py: evaluation checkpoint rule",
     "sys.path.insert(0, str(Path(__file__).resolve().parents[2] / \"scripts\"))\nimport checkpoints  # ../scripts/checkpoints.py"),
    ("import argparse\nimport json\nimport time\n", "import argparse\nimport json\nimport sys\nimport time\n"),
    ("CONTROL_DT = 0.04", "CONTROL_DT = 0.05"),
    ("base_kw = dict(cfg[\"agent\"][\"base\"]) if cfg else dict(inflate=0.05, shortcut_iters=20)",
     "base_kw = dict(cfg[\"agent\"][\"base\"]) if cfg else dict(inflate=0.017, shortcut_iters=20)"),
    ("    episodes, meta = testsets.load(args.set)\n", "    episodes, meta = testsets.load(args.set, ROOT / \"testsets\")\n"),
    ("        base_kw[\"max_checks\"] = int(budget)\n",
     "        if cfg:\n            base_kw[\"inflate\"] = 0.017  # MyCobot base (UR5e runs are probed zero-shot, SPEC M6)\n"
     "        base_kw[\"max_checks\"] = int(budget)\n"),
    ("(f\"{args.set}_first{args.limit}\" if args.limit else args.set)", "(f\"{args.set}_first{args.limit}\" if args.limit else args.set)"),
], "Evaluate methods on a frozen MyCobot test set (port of the UR5e script)")

port(UR / "scripts" / "train.py", HERE / "scripts" / "train.py", [
    ("from ur5rl.envs.residual import ResidualEnv\nfrom ur5rl.eval import testsets\nfrom ur5rl.envs.pose_env import EnvConfig, UR5ePoseEnv\n",
     "from mcrl.envs.residual import ResidualEnv\nfrom ur5rl.eval import testsets\nfrom mcrl.envs.pose_env import EnvConfig, MyCobotPoseEnv as UR5ePoseEnv\n"),
    ('    package = exp.get("package", "ur5rl")\n    env_dir = ROOT / "configs" / "env" if package == "ur5rl" else ROOT / "2-dof" / "configs" / "env"\n',
     '    package = "ur5rl"  # MyCobot lane: UR5e-style config handling, configs live under mycobot/\n    env_dir = ROOT / "configs" / "env"\n'),
    ('    if not str(exp["env"]).startswith("p5"):  # the torque env (P5) has no obstacle slots\n        env_cfg["n_obstacles"] = tuple(env_cfg.get("n_obstacles", (0, 0)))\n',
     '    env_cfg["n_obstacles"] = tuple(env_cfg.get("n_obstacles", (0, 0)))\n'),
    ("""    if cfg.get("package", "ur5rl") == "arm2dof":
        from arm2dof.envs.residual import ResidualArmEnv
        if str(cfg.get("stage", "")).startswith("p5"):  # torque control + gravity (MuJoCo)
            from arm2dof.envs.residual_torque import ResidualTorqueEnv
            from arm2dof.envs.torque_env import TorqueArmEnv, TorqueConfig
            return TorqueConfig, TorqueArmEnv, ResidualTorqueEnv
        from arm2dof.envs.kinematic_env import EnvConfig as ArmConfig, KinematicArmEnv
        return ArmConfig, KinematicArmEnv, ResidualArmEnv
    return EnvConfig, UR5ePoseEnv, ResidualEnv""", "    return EnvConfig, UR5ePoseEnv, ResidualEnv"),
    ("    val_eps = testsets.load(val_name)[0] if cfg.get(\"package\", \"ur5rl\") == \"ur5rl\" and testsets.exists(val_name) \\\n        else None",
     "    val_eps = testsets.load(val_name, ROOT / \"testsets\")[0] if testsets.exists(val_name, ROOT / \"testsets\") else None"),
    # crash-safe checkpoints (mcrl/callbacks.py): atomic alternating slots, fallback on load, saved every 10k steps
    ("from ur5rl.agents.residual import AlphaRamp\n", "from ur5rl.agents.residual import AlphaRamp\nfrom mcrl.callbacks import AtomicCheckpoint, pick_checkpoint\n"),
    ("""    ckpt = run_dir / "checkpoint"
    if (ckpt / "model.zip").exists():
        model = ALGOS[cfg["agent"]["algo"]].load(ckpt / "model.zip", env=venv, device="cpu")
        if (ckpt / "replay_buffer.pkl").exists():
            model.load_replay_buffer(ckpt / "replay_buffer.pkl")
        print(f"{rid}: resumed at {model.num_timesteps} steps", file=sys.stderr, flush=True)
    else:
        model = make_agent(cfg["agent"], venv, her=cfg["her"], seed=args.seed)""",
     """    model = pick_checkpoint(run_dir, ALGOS[cfg["agent"]["algo"]], venv)
    if model is None:
        model = make_agent(cfg["agent"], venv, her=cfg["her"], seed=args.seed)"""),
    ("ckpt_cb = CheckpointWithBuffer(run_dir,", "ckpt_cb = AtomicCheckpoint(run_dir,"),
    ('"--ckpt-every", type=int, default=50_000,', '"--ckpt-every", type=int, default=10_000,'),
], "Train one MyCobot run (port of the UR5e script); configs under mycobot/configs")

# the exp/env/agent yaml files and mcrl/robot.py, planners/ik.py are written by hand
(HERE / "mcrl" / "__init__.py").write_text("", encoding="utf-8")
(HERE / "mcrl" / "envs" / "__init__.py").write_text("", encoding="utf-8")
(HERE / "mcrl" / "planners" / "__init__.py").write_text("", encoding="utf-8")
