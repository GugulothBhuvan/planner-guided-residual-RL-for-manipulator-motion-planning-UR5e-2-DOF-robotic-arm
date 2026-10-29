"""Policy feature extractor that computes goal errors from the goal keys (SPEC §6.4).

The env's `observation` is goal-independent, so HER relabeling cannot leave stale goal features in it.
This extractor rebuilds the goal-relative features from achieved_goal / desired_goal inside the network,
so they are always consistent with whichever goal the sample carries.
"""

from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GoalErrorExtractor(BaseFeaturesExtractor):
    """[observation, achieved_goal, desired_goal, desired - achieved] -> flat features.

    desired - achieved is the TCP position error (3) and the r6d difference (6).
    """

    def __init__(self, observation_space: spaces.Dict):
        n_obs = observation_space["observation"].shape[0]
        n_goal = observation_space["desired_goal"].shape[0]
        super().__init__(observation_space, features_dim=n_obs + 3 * n_goal)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        ag, dg = obs["achieved_goal"], obs["desired_goal"]
        return torch.cat([obs["observation"], ag, dg, dg - ag], dim=1)
