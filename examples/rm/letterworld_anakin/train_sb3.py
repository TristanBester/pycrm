# ruff: noqa: D101, D102, D107
"""Baseline: numpy LetterWorld -> gym CrossProduct -> SB3 DQN (matched eval)."""

from __future__ import annotations

import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from examples.introduction.core.crossproduct import LetterWorldCrossProduct
from examples.introduction.core.ground import LetterWorld
from examples.introduction.core.label import LetterWorldLabellingFunction
from examples.introduction.core.machine import LetterWorldCountingRewardMachine

# Frozen hyperparameters from Task 3's spike (must match train_anakin.py).
HIDDEN = (64, 64)
GAMMA = 0.99
LR = 1e-3
BUFFER = 50_000
BATCH = 128
TARGET_UPDATE = 500
LEARNING_STARTS = 1_000
EXPLORATION_FRACTION = 0.2
EPS_START, EPS_END = 1.0, 0.05
MAX_STEPS = 100
EVAL_EVERY = 2_000
EVAL_EPISODES = 20

# The read-only LetterWorldCrossProduct declares a (3,) observation_space but its
# _get_obs actually emits a 5-vector [symbol_seen, row, col, u, c]; SB3 builds the
# policy from the declared space and then rejects the (5,) obs at predict time. We
# cannot edit the env, so make_env corrects the declared space to the true shape.
OBS_SHAPE = (5,)


def _make_cross_product() -> LetterWorldCrossProduct:
    """Build the numpy LetterWorld cross-product with a corrected obs space."""
    env = LetterWorldCrossProduct(
        ground_env=LetterWorld(),
        crm=LetterWorldCountingRewardMachine(),
        lf=LetterWorldLabellingFunction(),
        max_steps=MAX_STEPS,
    )
    env.observation_space = gym.spaces.Box(
        low=0, high=100, shape=OBS_SHAPE, dtype=np.int32
    )
    return env


def make_env(seed: int) -> Monitor:
    """Build the numpy LetterWorld cross-product env wrapped for episode stats."""
    env = _make_cross_product()
    env = Monitor(env)
    env.reset(seed=seed)
    return env


class _EvalRecorder(BaseCallback):
    def __init__(
        self, start: float, eval_every: int, eval_episodes: int, seed: int
    ) -> None:
        super().__init__()
        self.start = start
        self.eval_every = eval_every
        self.eval_episodes = eval_episodes
        self.seed = seed
        self.steps: list[int] = []
        self.wall: list[float] = []
        self.success: list[float] = []
        self.ret: list[float] = []
        self._next = eval_every

    def _run_eval(self) -> None:
        env = _make_cross_product()
        succ, rets = 0, []
        for e in range(self.eval_episodes):
            obs, _ = env.reset(seed=self.seed + 1000 + e)
            done = trunc = False
            total = 0.0
            while not (done or trunc):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, done, trunc, _ = env.step(int(action))
                total += float(r)
            succ += int(done and not trunc)
            rets.append(total)
        self.steps.append(int(self.num_timesteps))
        self.wall.append(time.perf_counter() - self.start)
        self.success.append(succ / self.eval_episodes)
        self.ret.append(float(np.mean(rets)))

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._run_eval()
            self._next += self.eval_every
        return True


def run_sb3_dqn(total_timesteps: int, seed: int, log_dir: str) -> dict:
    """Train SB3 DQN and return matched learning-curve + throughput metrics."""
    env = make_env(seed)
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=LR,
        buffer_size=BUFFER,
        batch_size=BATCH,
        gamma=GAMMA,
        target_update_interval=TARGET_UPDATE,
        learning_starts=LEARNING_STARTS,
        exploration_fraction=EXPLORATION_FRACTION,
        exploration_initial_eps=EPS_START,
        exploration_final_eps=EPS_END,
        policy_kwargs={"net_arch": list(HIDDEN)},
        seed=seed,
        verbose=0,
    )
    start = time.perf_counter()
    rec = _EvalRecorder(start, EVAL_EVERY, EVAL_EPISODES, seed)
    model.learn(total_timesteps=total_timesteps, callback=rec, progress_bar=False)
    elapsed = time.perf_counter() - start

    time_to_solve: float | None = None
    steps_to_solve: int | None = None
    for s, w, sr in zip(rec.steps, rec.wall, rec.success, strict=True):
        if sr >= 0.95:
            time_to_solve = w
            steps_to_solve = s
            break

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(log_dir) / f"sb3_dqn_seed{seed}_progress.csv"
    with csv_path.open("w") as fh:
        fh.write("step,wall_clock,success_rate,mean_return\n")
        for s, w, sr, mr in zip(
            rec.steps, rec.wall, rec.success, rec.ret, strict=True
        ):
            fh.write(f"{s},{w:.4f},{sr:.4f},{mr:.4f}\n")

    return {
        "backend": "sb3_dqn",
        "eval_steps": rec.steps,
        "eval_wall": rec.wall,
        "success_rate": rec.success,
        "mean_return": rec.ret,
        "steps_per_sec": total_timesteps / max(elapsed, 1e-9),
        "total_wall_clock": elapsed,
        "time_to_solve": time_to_solve,
        "steps_to_solve": steps_to_solve,
    }


if __name__ == "__main__":
    metrics = run_sb3_dqn(total_timesteps=150_000, seed=0, log_dir="results")
    print(
        f"SB3 DQN: {metrics['steps_per_sec']:.0f} steps/s, "
        f"{metrics['total_wall_clock']:.1f}s, "
        f"best success {max(metrics['success_rate']):.2f}, "
        f"solved@ {metrics['time_to_solve']} {metrics['steps_to_solve']}"
    )
