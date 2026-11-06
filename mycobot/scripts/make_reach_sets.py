"""Frozen reach episodes (SPEC M10): 100 test and 100 validation episodes per stage, sampled by the stage's env with fixed
seeds (test 7_000_000 + i, val 8_000_000 + i).  Output testsets/reach_<stage>_{test,val}.npz.

    python scripts/make_reach_sets.py r1 r2 r3 r4
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from mcrl.envs.reach_env import MyCobotReachEnv, ReachConfig  # noqa: E402
from mcrl.reachsets import save_set  # noqa: E402
from train_reach import stage_cfg  # noqa: E402

for stage in sys.argv[1:]:
    env = MyCobotReachEnv(ReachConfig(**stage_cfg(stage)))
    for which, base in (("test", 7_000_000), ("val", 8_000_000)):
        eps = []
        for i in range(100):
            env.reset(seed=base + i)  # a sampled episode (the env draws it from its RNG)
            eps.append(env.episode())
        save_set(ROOT / "testsets" / f"reach_{stage}_{which}.npz", eps)
        print(stage, which, len(eps))
