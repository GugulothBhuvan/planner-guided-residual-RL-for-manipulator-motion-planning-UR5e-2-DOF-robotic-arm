"""Set the S4 horizon on the frozen MyCobot sets without changing their episodes (SPEC M4, 2026-11-02 decision).

The sets were built with horizon 489 (rule: 1.25 x p99 of B2 durations). The cheap planner's paths are ~2x longer than
B2's (p90 ~754 steps), so 489 made it time-limited. The episodes stay as they are; only the horizon changes.
    python scripts/set_horizon.py 900
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
H = int(sys.argv[1])
for name in ("s4_test", "s4_val"):
    p = ROOT / "testsets" / f"{name}.npz"
    z = dict(np.load(p))
    z["horizon"] = np.array(H)
    np.savez_compressed(p, **z)
    print(name, "horizon ->", H, "episodes", len(z["seeds"]))
s = json.loads((ROOT / "testsets" / "summary.json").read_text())
for k in ("s4_test", "s4_val"):
    s[k]["horizon"] = H
s["s4_horizon"] = dict(rule="decision 2026-11-02: 900 steps (cheap-planner paths p90 ~754 steps); sets built at 489 by "
                            "max(200, ceil(1.25 * p99 B2 duration / 0.05 s))", horizon_steps=H, built_at=489)
(ROOT / "testsets" / "summary.json").write_text(json.dumps(s, indent=2))
y = ROOT / "configs" / "env" / "s4.yaml"
lines = y.read_text().splitlines()
y.write_text("\n".join(f"episode_length: {H}        # SPEC M4: 900 (decision 2026-11-02; was 489)" if l.startswith("episode_length:") else l
                       for l in lines) + "\n")
