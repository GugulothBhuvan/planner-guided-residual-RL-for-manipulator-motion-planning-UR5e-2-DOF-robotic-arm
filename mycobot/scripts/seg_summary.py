"""Summary of the segment-executor batch (results/seg/*_seg.json) next to the streaming runs (results/real/)."""
import json, glob, re, numpy as np
rows = {}
for tag, d, suf in (("segments", "results/seg", "_seg"), ("streaming", "results/real", "")):
    for m in ("b2", "base", "hybrid"):
        R = []
        for ep in range(10):
            f = f"{d}/real_s4_test_ep{ep}_{m}" + ("" if m == "b2" else "@3000") + suf + ".json"
            s = json.load(open(f))["summary"]
            R.append(s)
        rows[(tag, m)] = R
print(f"{'mode':10s} {'method':7s} {'ok':>3s} {'succ':>5s} {'coll':>4s} {'wall med (max) s':>17s} {'d_pos med (max) cm':>19s} {'theta med (max) deg':>20s} {'segments med':>12s}")
out = {}
for (tag, m), R in rows.items():
    w = [s["wall_s"] for s in R]; d = [s["d_pos_cm"] for s in R]; t = [s["theta_deg"] for s in R]
    ok = sum(s["status"] == "ok" for s in R); su = sum(bool(s.get("success")) for s in R); co = sum(bool(s.get("collision")) for s in R)
    sg = np.median([s.get("segments", np.nan) for s in R])
    print(f"{tag:10s} {m:7s} {ok:3d} {su:5d} {co:4d} {np.median(w):7.1f} ({max(w):5.1f})   {np.median(d):8.2f} ({max(d):5.2f})   {np.median(t):9.1f} ({max(t):5.1f})   {sg:12.0f}")
    out[f"{tag}_{m}"] = dict(ok=ok, success=su, collisions=co, wall_median=float(np.median(w)), wall_max=float(max(w)), d_pos_median=float(np.median(d)), d_pos_max=float(max(d)),
                             theta_median=float(np.median(t)), theta_max=float(max(t)))
json.dump(out, open("results/seg_summary.json", "w"), indent=1)
