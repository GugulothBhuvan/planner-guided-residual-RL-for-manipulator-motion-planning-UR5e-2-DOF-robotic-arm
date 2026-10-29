"""HER buffer that keeps only the goal-independent reward term (SPEC §6.7 decomposition).

SB3's HerReplayBuffer either passes empty infos to compute_reward (default), which silently drops the
safety terms, or deep-copies every info dict (copy_info_dict=True, ~340 B per transition here). This subclass
stores just info["r_indep"] in a float64 array (8 B per transition) and hands compute_reward that array.
"""

from __future__ import annotations

import numpy as np
from stable_baselines3 import HerReplayBuffer


class _RIndepStore:
    """Stands in for HerReplayBuffer.infos: `store[pos] = infos` keeps r_indep, `store[b, e]` returns floats."""

    def __init__(self, buffer_size, n_envs):
        self.values = np.zeros((buffer_size, n_envs), dtype=np.float64)

    def __setitem__(self, pos, infos):
        self.values[pos] = [info["r_indep"] for info in infos]

    def __getitem__(self, idx):
        return self.values[idx]


class SafetyHerReplayBuffer(HerReplayBuffer):
    def __init__(self, *args, **kwargs):
        kwargs["copy_info_dict"] = True  # route infos through self.infos, which is replaced below
        super().__init__(*args, **kwargs)
        self.infos = _RIndepStore(self.buffer_size, self.n_envs)
