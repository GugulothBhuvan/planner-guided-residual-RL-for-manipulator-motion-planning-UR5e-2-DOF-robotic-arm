"""GoalCurriculum callback (SPEC §6.5): raise on rolling success > threshold, cap, persist, resume."""

from types import SimpleNamespace

from ur5rl.agents.callbacks import GoalCurriculum


class StubVecEnv:
    def __init__(self):
        self.calls = []

    def env_method(self, name, *args):
        self.calls.append((name, args))


def make(tmp_path, **kw):
    cb = GoalCurriculum(tmp_path, start=0.6, step=0.3, cap=1.2, threshold=0.8, window=10, **kw)
    env = StubVecEnv()
    # SB3 callbacks reach the env and logger through the model
    cb.model = SimpleNamespace(get_env=lambda: env, logger=SimpleNamespace(record=lambda *a, **k: None))
    cb.num_timesteps = 0
    return cb, env


def episodes(cb, successes):
    for s in successes:
        cb.locals = dict(dones=[True], infos=[dict(is_success=s)])
        cb._on_step()


def test_raises_only_above_threshold_and_caps(tmp_path):
    cb, env = make(tmp_path)
    cb._on_training_start()
    assert env.calls[-1] == ("set_curriculum_level", (0.6,))
    episodes(cb, [False] * 2 + [True] * 8)  # 80%: not above threshold
    assert cb.level == 0.6
    episodes(cb, [True])  # window is now F + 9 T = 90% -> raise
    assert abs(cb.level - 0.9) < 1e-12 and env.calls[-1] == ("set_curriculum_level", (cb.level,))
    episodes(cb, [True] * 9)  # the window restarts after a raise: 9 episodes are not enough
    assert abs(cb.level - 0.9) < 1e-12
    episodes(cb, [True])
    assert abs(cb.level - 1.2) < 1e-12
    episodes(cb, [True] * 30)
    assert abs(cb.level - 1.2) < 1e-12  # capped


def test_full_range_episodes_do_not_count(tmp_path):
    cb, _ = make(tmp_path)
    for _ in range(50):  # full-range failures must not hold the level back
        cb.locals = dict(dones=[True], infos=[dict(is_success=False, curriculum_goal=False)])
        cb._on_step()
    for _ in range(10):
        cb.locals = dict(dones=[True], infos=[dict(is_success=True, curriculum_goal=True)])
        cb._on_step()
    assert abs(cb.level - 0.9) < 1e-12


def test_level_persists_for_resume(tmp_path):
    cb, _ = make(tmp_path)
    episodes(cb, [True] * 10)
    resumed, env = make(tmp_path)
    resumed._on_training_start()
    assert abs(resumed.level - 0.9) < 1e-12 and env.calls[-1] == ("set_curriculum_level", (resumed.level,))
