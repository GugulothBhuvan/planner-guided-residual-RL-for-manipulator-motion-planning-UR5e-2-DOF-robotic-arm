"""Day-1 throughput gate (SPEC §11): end-to-end SB3 SAC on the S4 env must reach >= 100 steps/s.

"steps" = environment transitions (summed over the parallel envs), which is what the §11 budget counts.
The rate is measured after learning_starts, i.e. with gradient updates running.

    python scripts/bench_throughput.py --mode dense --threads 1
    python scripts/bench_throughput.py --mode her --threads 1
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import torch
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from ur5rl.agents.callbacks import peak_rss_mb
from ur5rl.envs.pose_env import EnvConfig, UR5ePoseEnv


class Timer(BaseCallback):
    def __init__(self, start_at):
        super().__init__()
        self.start_at, self.t_start, self.n_start = start_at, None, None

    def _on_step(self):
        if self.t_start is None and self.num_timesteps >= self.start_at:
            self.t_start, self.n_start = time.perf_counter(), self.num_timesteps
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["dense", "her"], default="dense")
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--vec", choices=["dummy", "subproc"], default="dummy")
    p.add_argument("--grad-per-transition", type=float, default=0.5)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--frame-skip", type=int, default=20)
    p.add_argument("--timestep", type=float, default=0.002)
    p.add_argument("--net", default="256,256,256")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    cfg = dict(n_obstacles=(1, 4), episode_length=200, reward="dense" if args.mode == "dense" else "sparse",
               frame_skip=args.frame_skip, timestep=args.timestep)
    env = make_vec_env(UR5ePoseEnv, n_envs=args.n_envs, seed=0, env_kwargs=dict(config=EnvConfig(**cfg)),
                       vec_env_cls=SubprocVecEnv if args.vec == "subproc" else DummyVecEnv)
    # train_freq counts vec-env steps (n_envs transitions each), so gradient_steps is scaled by n_envs.
    grad_steps = max(1, round(args.grad_per_transition * args.n_envs))
    learning_starts = 1000
    kw = dict(learning_rate=3e-4, buffer_size=200_000, batch_size=args.batch, tau=0.05,
              gamma=0.99 if args.mode == "dense" else 0.95, train_freq=1, gradient_steps=grad_steps,
              learning_starts=learning_starts, device="cpu", seed=0, verbose=0,
              policy_kwargs=dict(net_arch=[int(x) for x in args.net.split(",")]))
    if args.mode == "her":
        kw.update(replay_buffer_class=HerReplayBuffer,
                  replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future",
                                            copy_info_dict=True))  # SB3 drops infos otherwise (§6.7)
    model = SAC("MultiInputPolicy", env, **kw)

    timer = Timer(start_at=learning_starts + 1000)
    t0 = time.perf_counter()
    model.learn(total_timesteps=args.steps, callback=timer)
    t_end = time.perf_counter()
    rate_train = (model.num_timesteps - timer.n_start) / (t_end - timer.t_start)
    rate_total = model.num_timesteps / (t_end - t0)
    env.close()

    result = dict(mode=args.mode, n_envs=args.n_envs, vec=args.vec, threads=args.threads,
                  grad_steps_per_vec_step=grad_steps, grad_per_transition=grad_steps / args.n_envs,
                  batch=args.batch, net=args.net, frame_skip=args.frame_skip, timestep=args.timestep,
                  steps=model.num_timesteps, steps_per_s_training=round(rate_train, 1),
                  steps_per_s_overall=round(rate_total, 1), gate_100=rate_train >= 100,
                  machine=platform.processor(), cpus=torch.get_num_threads(), peak_rss_mb=round(peak_rss_mb()))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
