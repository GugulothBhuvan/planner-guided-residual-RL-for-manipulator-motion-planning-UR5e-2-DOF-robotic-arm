"""Zip the repo (code, configs, assets, lockfile; no .venv/results) for Colab/Kaggle: dist/ur5rl_bundle.zip.

    python scripts/make_bundle.py
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["src", "scripts", "configs", "assets", "testsets", "tests", "notebooks", "2-dof", "pyproject.toml", "requirements.lock",
           "README.md", "SPEC.md"]
SKIP_PARTS = {"__pycache__", ".pytest_cache"}


def main():
    out = ROOT / "dist" / "ur5rl_bundle.zip"
    out.parent.mkdir(exist_ok=True)
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for item in INCLUDE:
            path = ROOT / item
            files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
            for f in files:
                rel = f.relative_to(ROOT)
                if SKIP_PARTS & set(rel.parts) or any(part.endswith(".egg-info") for part in rel.parts):
                    continue
                z.write(f, rel.as_posix())
                n += 1
    print(f"wrote {out.relative_to(ROOT)}: {n} files, {out.stat().st_size / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
