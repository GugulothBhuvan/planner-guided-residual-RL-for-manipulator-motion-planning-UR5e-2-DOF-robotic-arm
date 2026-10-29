"""Exclusive lock file for read-merge-write updates of a shared results file (evaluation summary.json)."""

from __future__ import annotations

import os
import time
from pathlib import Path


class SummaryLock:
    """Lock around a summary.json read-merge-write: parallel evaluations would otherwise drop each other's entries."""

    def __init__(self, path):
        self.path = Path(str(path) + ".lock")

    def __enter__(self):
        t0 = time.time()
        while True:
            try:
                os.close(os.open(self.path, os.O_CREAT | os.O_EXCL))
                return self
            except FileExistsError:
                if time.time() - t0 > 120:  # a holder that died leaves the file behind; a merge takes milliseconds
                    self.path.unlink(missing_ok=True)
                time.sleep(0.2)

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)
