# ruff: noqa: D101, D102, D107
"""Baseline: numpy PuckWorld -> gym CrossProduct -> SB3 DQN."""

from __future__ import annotations

import time
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from examples.rm.discrete.core.crossproduct import PuckWorldCrossProduct
from examples.rm.discrete.core.ground import PuckWorld
from examples.rm.discrete.core.label import PuckWorldLabellingFunction
from examples.rm.discrete.core.machine import PuckWorldRewardMachine

HIDDEN = (256, 256)
GAMMA = 0.99
LR = 2.5e-4
BUFFER = 100_000
BATCH = 128
TARGET_UPDATE = 1_000
LEARNING_STARTS = 1_000
EXPLORATION_FRACTION = 0.2
MAX_STEPS = 1_000
EVAL_EVERY = 5_000


def make_env(seed: int) -> Monitor:
    """Build the numpy PuckWorld cross-product env wrapped for episode stats."""
    env = PuckWorldCrossProduct(
        ground_env=PuckWorld(),
        machine=PuckWorldRewardMachine(),
        lf=PuckWorldLabellingFunction(),
        max_steps=MAX_STEPS,
    )
    env.reset(seed=seed)
    return Monitor(env)


class _Recorder(BaseCallback):
    def __init__(self, start: float) -> None:
        super().__init__()
        self.start = start
        self.returns: list[float] = []
        self.steps: list[int] = []
        self.wall: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                self.returns.append(float(ep["r"]))
                self.steps.append(int(self.num_timesteps))
                self.wall.append(time.perf_counter() - self.start)
        return True


def run_sb3_dqn(total_timesteps: int, seed: int, log_dir: str) -> dict:
    """Train SB3 DQN and return learning-curve + throughput metrics."""
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
        policy_kwargs={"net_arch": list(HIDDEN)},
        seed=seed,
        verbose=0,
    )
    start = time.perf_counter()
    rec = _Recorder(start)
    model.learn(total_timesteps=total_timesteps, callback=rec, progress_bar=False)
    elapsed = time.perf_counter() - start

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    csv = Path(log_dir) / "sb3_progress.csv"
    with csv.open("w") as fh:
        fh.write("step,wall_clock,return\n")
        for s, w, r in zip(rec.steps, rec.wall, rec.returns, strict=True):
            fh.write(f"{s},{w:.4f},{r:.4f}\n")

    return {
        "backend": "sb3_dqn",
        "returns": rec.returns,
        "steps": rec.steps,
        "wall_clock": rec.wall,
        "steps_per_sec": total_timesteps / max(elapsed, 1e-9),
        "total_wall_clock": elapsed,
    }


if __name__ == "__main__":
    metrics = run_sb3_dqn(total_timesteps=50_000, seed=0, log_dir="results")
    print(
        f"SB3 DQN: {metrics['steps_per_sec']:.0f} steps/s, "
        f"{metrics['total_wall_clock']:.1f}s, "
        f"{len(metrics['returns'])} episodes"
    )
