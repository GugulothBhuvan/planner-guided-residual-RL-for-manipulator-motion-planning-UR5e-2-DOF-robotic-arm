"""PE8 evaluation (companion SPEC §8, §9): B5, B5', SAC+HER and residual-on-B5' on the exact and +-20% mass plants.

Test set (§9.2: 300 frozen episodes; seeds decided 2026-10-29): episode i is the one TorqueArmEnv draws at reset seed
5_400_000 + i, frozen to 2-dof/testsets/p5a_test.npz on first use (disjoint from the P1-P3 test seeds 5.0M-5.3M and
from training validation, 1_000_000 + i); the +-20% copies replay the same 300 episodes. Plants: mass_factor 1.0 (exact), 0.8 and 1.2 (links' mass and inertia scaled); every controller and
the residual's base keep the nominal model. One success rule for all methods: the end effector within eps (1 cm)
and |q'| < 0.1 rad/s for success_hold (5) consecutive steps within the 307-step horizon (the dense env's terminal
condition, so episodes end there). Also recorded: settling time (time to success), energy sum(tau^2 dt), peak |tau|
(§9.1). Trained runs: SAC+HER at best validation; residual at the UR5e SPEC §9.2 primary
and final checkpoints (for 300k-step runs both are the 300k model: the actor trains from ~202k).

    python 2-dof/scripts/eval_pe8.py   ->  results/eval/2dof/p5a_test/*.jsonl, summary.json; results/tables/pe8.md
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from arm2dof.envs.residual_torque import ResidualTorqueEnv
from arm2dof.envs.torque_env import TorqueArmEnv, TorqueConfig

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import checkpoints  # noqa: E402  UR5e scripts/checkpoints.py
from e5_table import seed_ci, wilson  # noqa: E402  same intervals as the E5 headline table (SPEC §9.3)

SEED0, N = 5_400_000, 300
TEST = ROOT / "2-dof" / "testsets" / "p5a_test.npz"
PLANTS = (1.0, 0.8, 1.2)
OUT = ROOT / "results" / "eval" / "2dof" / "p5a_test"


def stage_cfg(mass_factor):
    cfg = yaml.safe_load((ROOT / "2-dof" / "configs" / "env" / "p5a.yaml").read_text(encoding="utf-8"))
    return TorqueConfig(**dict(cfg, reward="dense", mass_factor=mass_factor))  # dense: terminate on the hold rule


def load(run, which):
    from stable_baselines3.common.vec_env import DummyVecEnv
    from ur5rl.agents.factory import ALGOS
    cfg = json.loads((run / "config.json").read_text())["config"]
    residual = cfg["agent"]["algo"] == "residual_sac"
    make = (lambda: ResidualTorqueEnv(TorqueArmEnv(TorqueConfig(**cfg["env"])), **cfg["agent"]["base"])) if residual \
        else (lambda: TorqueArmEnv(TorqueConfig(**cfg["env"])))
    path, _, note = checkpoints.select(run, which)
    print(f"{run.name}: {which} checkpoint {path.name} ({note})", flush=True)
    return ALGOS[cfg["agent"]["algo"]].load(path, env=DummyVecEnv([make]), device="cpu"), cfg


def episodes():
    """The frozen P5a test episodes (created from the reset seeds on first use)."""
    if not TEST.exists():
        env = TorqueArmEnv(stage_cfg(1.0))
        eps = []
        for i in range(N):
            env.reset(seed=SEED0 + i)
            eps.append(env.episode())
        np.savez(TEST, q_start=np.array([e["q_start"] for e in eps]), q_goal=np.array([e["q_goal"] for e in eps]),
                 seeds=SEED0 + np.arange(N))
    z = np.load(TEST)
    return [dict(q_start=a, q_goal=b) for a, b in zip(z["q_start"], z["q_goal"])]


def rollout(env, act, ep):
    obs, _ = env.reset(options={"episode": ep})
    base, energy, peak, t = env.unwrapped, 0.0, 0.0, 0
    while True:
        obs, _, term, trunc, info = env.step(act(obs))
        tau = base.data.ctrl
        energy += float((tau ** 2).sum()) * base.dt
        peak = max(peak, float(np.abs(tau).max()))
        t += 1
        if term or trunc:
            return dict(success=bool(term), d_pos_cm=100 * info["d_pos"], qd_end=info["qd_norm"],
                        time_to_success_s=t * base.dt if term else None, energy=energy, peak_torque=peak)


def evaluate(method, mass_factor, run=None, which="best"):
    model, cfg = load(run, which) if run else (None, None)
    plant = TorqueArmEnv(stage_cfg(mass_factor))
    if method in ("b5", "b5p"):
        env = ResidualTorqueEnv(plant, alpha=0.0, kind="ct" if method == "b5" else "pdg")
    elif cfg["agent"]["algo"] == "residual_sac":
        env = ResidualTorqueEnv(plant, alpha=cfg["agent"]["alpha_max"], **cfg["agent"]["base"])
    else:
        env = plant
    act = (lambda o: np.zeros(2)) if model is None else (lambda o: model.predict(o, deterministic=True)[0])
    return [rollout(env, act, ep) for ep in episodes()]


def jobs():
    out = [("b5", None, "-"), ("b5p", None, "-")]
    for d in sorted((ROOT / "results" / "runs" / "pe8").glob("*_s*")):
        if (d / "done.json").exists():
            residual = d.name.startswith("residual")
            out += [(d.name, d, c) for c in (("primary", "final") if residual else ("best",))]
    return out


def table(results):
    names = {"b5": "B5 (computed torque)", "b5p": "B5' (PD + gravity)", "residual": "residual on B5'",
             "sac_her": "SAC+HER (torque)"}
    md = [f"# PE8: P5a torque reaching, exact and +-20% mass plants ({N} episodes per seed)", "",
          "Success in %. Trained arms: mean over seeds [95% bootstrap CI over seeds] (pooled Wilson 95%, episode-level "
          "only). Baselines: one deterministic evaluation (Wilson 95%). Final error in cm (mean). Robustness gap = "
          "success(exact) - worst perturbed success, in points.", "",
          "| Method | Checkpoint | Seeds | " + " | ".join(f"Success, mass x{m}" for m in PLANTS) + " | "
          + " | ".join(f"d_pos x{m}" for m in PLANTS) + " | Robustness gap |",
          "|---|---|---|" + "---|" * (2 * len(PLANTS) + 1)]
    groups: dict = {}
    for (name, ckpt, m), rows in results.items():
        arm = name if name in ("b5", "b5p") else name.split("_")[0] if name.startswith("residual") else "sac_her"
        groups.setdefault((arm, ckpt), {}).setdefault(m, []).append(rows)
    for (arm, ckpt), by_m in sorted(groups.items()):
        cells, errs, means = [], [], {}
        for m in PLANTS:
            seeds = by_m.get(m, [])
            if not seeds:
                cells.append("-"), errs.append("-")
                continue
            per = [float(np.mean([r["success"] for r in rows])) for rows in seeds]
            pooled = [r["success"] for rows in seeds for r in rows]
            lo, hi = wilson(sum(pooled), len(pooled))
            means[m] = float(np.mean(per))
            c = f"{100 * np.mean(per):.1f}"
            if len(seeds) > 1:
                slo, shi = seed_ci(per)
                c += f" [{100 * slo:.1f}, {100 * shi:.1f}]"
            cells.append(c + f" ({100 * lo:.1f}–{100 * hi:.1f})")
            errs.append(f"{np.mean([r['d_pos_cm'] for rows in seeds for r in rows]):.2f}")
        n = len(by_m.get(1.0, []))
        md.append(f"| {names.get(arm, arm)} | {ckpt} | {n if arm not in ('b5', 'b5p') else '-'} | "
                  + " | ".join(cells) + " | " + " | ".join(errs) + " | "
                  + (f"{100 * (means[1.0] - min(means[m] for m in PLANTS[1:])):.1f}" if len(means) == len(PLANTS) else "-")
                  + " |")
    return "\n".join(md) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, run, ckpt in jobs():
        for m in PLANTS:
            f = OUT / f"{name}__{ckpt}__m{m}.jsonl"
            if not f.exists():
                t0 = time.perf_counter()
                rows = evaluate(name, m, run, ckpt) if run else evaluate(name, m)
                f.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
                print(f"{name} {ckpt} x{m}: success {np.mean([r['success'] for r in rows]):.3f} "
                      f"({time.perf_counter() - t0:.0f} s)", flush=True)
            results[(name, ckpt, m)] = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    (ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
    md = table(results)
    (ROOT / "results" / "tables" / "pe8.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
