"""One-cell, resumable Colab/Kaggle notebook that runs a list of (experiment, run, seed) jobs one at a time (SPEC §11).

Colab: checkpoints go to MyDrive/mtp/runs; train.py skips finished runs and resumes from the last checkpoint, so after
a disconnect the cell is just run again. Kaggle (detected by /kaggle): the bundle is an attached dataset, runs are
written to /kaggle/working/runs (the version's output), training stops before Kaggle's 12 h session limit, and a later
version with the earlier output attached resumes from it. Run ids are computed here exactly as train.py does (buffer
cap included) and printed in the header, so the folders merge into results/runs/<exp>/ on the laptop.

    python scripts/make_colab_notebook.py e5_s1s2      ->  notebooks/colab_e5_s1s2.ipynb
    python scripts/make_colab_notebook.py e5_sparse    ->  notebooks/colab_e5_sparse.ipynb
    python scripts/make_colab_notebook.py e5b_s1s2     ->  notebooks/colab_e5b_s1s2.ipynb
    python scripts/make_colab_notebook.py pe_s234      ->  notebooks/colab_pe_s234.ipynb  (2-DOF, parallel jobs)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train import resolve, run_id  # noqa: E402

E5, SPARSE = "configs/exp/e5.yaml", "configs/exp/e5_residual_sparse.yaml"
PE = "2-dof/configs/exp/{}.yaml"
# (jobs in order, header). Dense E5 seeds 1 and 2 (s0 on the laptop; marker results/runs/e5_offloaded_s1s2), and the
# sparse-residual arm's seeds 1 and 2 (§7.3; s0 on the laptop; marker results/runs/e5_sparse_offloaded_s1s2).
PLANS = {
    "e5_s1s2": ([(E5, "residual", 1), (E5, "residual", 2), (E5, "sac_her", 1), (E5, "sac_her", 2)],
                "SPEC §8 E5 (S4, 1M steps each): dense seeds 1 and 2, residual SAC first, then pure SAC+HER. Seed 0\n"
                "runs on the laptop (results/runs/e5_offloaded_s1s2 tells it to skip these)."),
    "e5_sparse": ([(SPARSE, "residual_sparse", s) for s in (1, 2)],
                  "SPEC §7.3 sparse-residual E5 arm (residual SAC, sparse reward, no HER, absorbing collision penalty,\n"
                  "same warm-up / ramp / w_res), seeds 1 and 2 x 1M steps. Seed 0 runs on the laptop\n"
                  "(results/runs/e5_sparse_offloaded_s1s2)."),
    # E5b (SPEC §7.3, pre-registered 2026-10-29): integrated-offset residual, seeds 1 and 2 (s0 on the laptop; marker
    # results/runs/e5b_offloaded_s1s2). Both run in parallel at 1 torch thread each when the VM has the cores.
    "e5b_s1s2": ([("configs/exp/e5b.yaml", "residual_offset", s) for s in (1, 2)],
                 "SPEC §7.3 E5b: residual SAC with an integrated joint offset (sparse reward, no HER, absorbing collision\n"
                 "penalty, E5's warm-up / ramp / w_res), seeds 1 and 2 x 1M steps. Seed 0 runs on the laptop\n"
                 "(results/runs/e5b_offloaded_s1s2).",
                 "parallel"),
    # 2-DOF (companion SPEC §8, decision 2026-10-29): seeds 2-4 of PE8 (300k steps, longest first), PE3 and PE4
    # (150k); seeds 0-1 on the laptop (markers results/runs/pe34_offloaded_s234, pe8_offloaded_s234). 1 torch thread
    # per job, as on the laptop; jobs run in parallel, one per CPU core.
    "pe_s234": ([(PE.format(e), r, s) for e in ("pe8", "pe3", "pe4") for s in (2, 3, 4) for r in ("residual", "sac_her")],
                "2-DOF companion SPEC §8: PE8 (P5a torque), PE3 (P2) and PE4 (P3), seeds 2-4, residual SAC and SAC+HER.\n"
                "Seeds 0-1 run on the laptop (results/runs/pe34_offloaded_s234, pe8_offloaded_s234). Jobs run in\n"
                "parallel, one per CPU core, 1 torch thread each (a CPU or TPU VM; no accelerator is used).",
                "parallel"),
}

CELL = """\
{header}
# Needs ur5rl_bundle.zip built {date} or later. CPU runtime. Jobs, in this order:
{ids}
# Colab: bundle at MyDrive/mtp/ur5rl_bundle.zip. Checkpoints (model + replay buffer, ~0.85 GB per run) go to
#   MyDrive/mtp/runs every 50k steps (needs ~{gb:.1f} GB free on Drive). After a disconnect, run this cell again: it
#   resumes (<= 50k steps lost). When done, copy MyDrive/mtp/runs/{exp}/<run id> into results/runs/{exp}/ on the laptop.
# Kaggle: attach the bundle as a dataset, Internet on, run with "Save & Run All (Commit)". Runs go to this version's
#   output (runs/{exp}/<run id>); training stops before the 12 h limit. To continue, attach that output as an input
#   of a new version and commit again: it resumes from it.
JOBS = {jobs!r}
import glob, os, shutil, subprocess, time
T0 = time.time()
sh = lambda c: subprocess.run(c, shell=True, check=True)
if os.path.exists('/kaggle'):
    W, OUT, DEADLINE = '/tmp/ur5rl', '/kaggle/working/runs', T0 + 11.4 * 3600  # Kaggle stops a session at 12 h
    zips = glob.glob('/kaggle/input/**/ur5rl_bundle.zip', recursive=True)
    if zips:
        sh(f'mkdir -p {{W}} && unzip -q -o {{zips[0]}} -d {{W}}')
    else:  # Kaggle may unpack an uploaded zip into a folder
        root = os.path.dirname(os.path.dirname(glob.glob('/kaggle/input/**/scripts/train.py', recursive=True)[0]))
        shutil.copytree(root, W, dirs_exist_ok=True)
    for prev in glob.glob('/kaggle/input/**/runs/*/*_s[0-9]', recursive=True):  # an earlier version's output: resume
        dst = os.path.join(OUT, *prev.rstrip('/').split('/')[-2:])
        if not os.path.exists(dst):
            shutil.copytree(prev, dst)
            print('resuming from', prev)
else:
    from google.colab import drive
    drive.mount('/content/drive')
    W, OUT, DEADLINE = '/content/ur5rl', '/content/drive/MyDrive/mtp/runs', None
    sh(f'mkdir -p {{W}} && unzip -q -o /content/drive/MyDrive/mtp/ur5rl_bundle.zip -d {{W}}')  # every time: current code
if not os.path.exists('/tmp/.deps_ok'):  # fresh VM: install once
    sh('pip install -q --index-url https://download.pytorch.org/whl/cpu torch==2.14.1')
    sh('pip install -q mujoco==3.17.0 gymnasium==1.4.0 stable-baselines3==2.9.2 numpy==2.4.6 scipy==1.17.1 '
       'PyYAML==6.0.3 psutil==7.2.2 tensorboard==2.21.0')
    sh(f'pip install -q -e {{W}} --no-deps --ignore-requires-python')
    if os.path.exists(f'{{W}}/2-dof/pyproject.toml'):
        sh(f'pip install -q -e {{W}}/2-dof --no-deps --ignore-requires-python')
    sh('touch /tmp/.deps_ok')
WORKERS = {workers}
THREADS = 2 if WORKERS == 1 else 1


def job(j):
    exp, run, seed = j
    left = None if DEADLINE is None else int(DEADLINE - time.time())
    if left is not None and left < 900:
        print(f'skip {{run}} s{{seed}}: too close to the Kaggle time limit (commit a new version with this output attached)')
        return
    print(f'=== {{exp}} {{run}} seed {{seed}}', flush=True)
    limit = f'timeout {{left}}s ' if left else ''
    subprocess.run(f'cd {{W}} && {{limit}}python scripts/train.py --exp {{exp}} --run {{run}} --seed {{seed}} --threads {{THREADS}} '
                   f'--out {{OUT}} --ckpt-every 50000 --keep-full-at 2>&1 '
                   f'| grep --line-buffered -E "val]|resumed|done|Error|Traceback" | sed -u "s/^/[{{run}} s{{seed}}] /"',
                   shell=True, check=DEADLINE is None)  # Kaggle: a run stopped by the time limit must not fail the cell


if WORKERS == 1:
    for j in JOBS:
        job(j)
else:
    from concurrent.futures import ThreadPoolExecutor
    n = max(1, min(len(JOBS), (os.cpu_count() or 2)))
    print(f'{{os.cpu_count()}} CPUs: {{n}} jobs in parallel', flush=True)
    with ThreadPoolExecutor(n) as pool:
        list(pool.map(job, JOBS))
print('finished (or stopped at the time limit); finished runs contain done.json')
"""


def expected_id(exp, run, seed):
    cfg = resolve(str(ROOT / exp), run)
    cfg["agent"]["buffer_size"] = min(int(cfg["agent"]["buffer_size"]), cfg["steps"])  # as train.py
    return cfg["exp"], run_id(cfg, seed)


def main():
    import time
    p = argparse.ArgumentParser()
    p.add_argument("plan", choices=sorted(PLANS))
    args = p.parse_args()
    jobs, header, *mode = PLANS[args.plan]
    ids = [expected_id(*j) for j in jobs]
    exp = "|".join(dict.fromkeys(e for e, _ in ids))  # e.g. "pe8|pe3|pe4" for a multi-experiment plan
    src = CELL.format(header="\n".join("# " + line for line in header.splitlines()), date=time.strftime("%Y-%m-%d"),
                      ids="\n".join(f"#   {i + 1}. {rid}" for i, (_, rid) in enumerate(ids)),
                      gb=0.9 * len(jobs) + 0.5, exp=exp, jobs=jobs, workers=0 if mode else 1)
    nb = dict(cells=[dict(cell_type="code", execution_count=None, metadata={}, outputs=[],
                          source=src.splitlines(keepends=True))],
              metadata={"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
              nbformat=4, nbformat_minor=5)
    out = ROOT / "notebooks" / f"colab_{args.plan}.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}:", ", ".join(rid for _, rid in ids))


if __name__ == "__main__":
    main()
