import argparse
import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from examples.crm.tabular.core.ground import OfficeWorld
from examples.crm.tabular.core.label import OfficeWorldLabellingFunction, Symbol
from examples.crm.tabular.core.machine import OfficeWorldCountingRewardMachine
from pycrm.crossproduct import CrossProduct

ALGORITHMS = ("dqn", "cdqn")
BACKENDS = ("sb3", "sbx")


class FixedMailOfficeWorld(OfficeWorld):
    """OfficeWorld variant with a fixed high mail count every episode."""

    def __init__(self, mail_count: int, start_row: int, start_col: int) -> None:
        """Initialize the ground environment."""
        super().__init__(max_n=mail_count, start_row=start_row, start_col=start_col)
        self.mail_count = mail_count

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset and force the task to require the maximum mail count."""
        obs, info = super().reset(seed=seed, options=options)
        self.total_mail = self.mail_count
        self.mail_collected = 0
        self.state[2] = 0
        return obs.copy(), info


class HighMailCountingRewardMachine(OfficeWorldCountingRewardMachine):
    """OfficeWorld CRM with counterfactual samples up to the chosen mail count."""

    def __init__(self, mail_count: int) -> None:
        """Initialize the counting reward machine."""
        self.mail_count = mail_count
        super().__init__()

    def _get_possible_counter_configurations(self) -> list[tuple[int, ...]]:
        """Return reachable high-mail counter configurations."""
        configs = set()
        for value in range(self.mail_count + 1):
            configs.add((value, value))
            configs.add((value, self.mail_count))
            configs.add((0, value))
        return sorted(configs)

    def _get_reward_transition_function(self) -> dict:
        """Return shaped rewards for the high-mail function approximation task."""
        return {
            0: {
                "M / (-,-)": 1.0,
                "E / (-,-)": -0.05,
                "C / (-,-)": -0.05,
                "P / (-,-)": -0.05,
                "D / (-,-)": -10.0,
                "/ (-,-)": -0.05,
            },
            1: {
                "M / (-,-)": -0.05,
                "E / (-,-)": -10.0,
                "C / (-,-)": 1.0,
                "P / (-,-)": -0.05,
                "D / (-,-)": -10.0,
                "/ (NZ,-)": -0.05,
                "/ (Z,-)": -0.05,
            },
            2: {
                "M / (-,-)": -0.05,
                "E / (-,-)": -0.05,
                "C / (-,-)": -0.05,
                "P / (-,-)": 1.0,
                "D / (-,-)": -10.0,
                "/ (-,NZ)": -0.05,
                "/ (-,Z)": 10.0,
            },
        }


class OfficeWorldDqnCrossProduct(CrossProduct[np.ndarray, np.ndarray, int, None]):
    """Normalized OfficeWorld cross-product for neural DQN agents."""

    def __init__(
        self, *, mail_count: int, max_steps: int, start_row: int, start_col: int
    ) -> None:
        """Initialize the cross-product task."""
        super().__init__(
            ground_env=FixedMailOfficeWorld(
                mail_count=mail_count,
                start_row=start_row,
                start_col=start_col,
            ),
            machine=HighMailCountingRewardMachine(mail_count=mail_count),
            lf=OfficeWorldLabellingFunction(),
            max_steps=max_steps,
        )
        self.mail_count = mail_count
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(9,),
            dtype=np.float32,
        )
        self.action_space = self.ground_env.action_space

    def _get_obs(
        self, ground_obs: np.ndarray, u: int, c: tuple[int, ...]
    ) -> np.ndarray:
        """Return normalized ground state, one-hot CRM state, and counters."""
        u_enc = self.crm.encode_machine_state(u).astype(np.float32)
        counters = np.array(c, dtype=np.float32) / float(max(self.mail_count, 1))
        return np.concatenate(
            (
                np.array(
                    [
                        float(ground_obs[0]) / 12.0,
                        float(ground_obs[1]) / 16.0,
                        float(ground_obs[2]),
                    ],
                    dtype=np.float32,
                ),
                u_enc,
                counters,
            )
        )

    def to_ground_obs(self, obs: np.ndarray) -> np.ndarray:
        """Recover [row, col, mail_empty] from a normalized observation."""
        row = int(np.rint(float(obs[0]) * 12.0))
        col = int(np.rint(float(obs[1]) * 16.0))
        empty = int(np.rint(float(obs[2])))
        return np.array(
            [
                np.clip(row, 0, 12),
                np.clip(col, 0, 16),
                np.clip(empty, 0, 1),
            ],
            dtype=np.int32,
        )

    def generate_counterfactual_experience(
        self, ground_obs: np.ndarray, action: int, next_ground_obs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate event-aware counterfactual replay for high-mail OfficeWorld."""
        (
            obs_buffer,
            action_buffer,
            obs_next_buffer,
            reward_buffer,
            done_buffer,
            info_buffer,
        ) = ([] for _ in range(6))
        props = self.lf(ground_obs, action, next_ground_obs)

        for u_i, configurations in self._counterfactual_configurations(props).items():
            for c_i in configurations:
                try:
                    u_j, c_j, rf_j = self.crm.transition(u_i, c_i, props)
                except ValueError:
                    continue

                obs_buffer.append(self._get_obs(ground_obs, u_i, c_i))
                action_buffer.append(action)
                obs_next_buffer.append(self._get_obs(next_ground_obs, u_j, c_j))
                reward_buffer.append(rf_j(ground_obs, action, next_ground_obs))
                done_buffer.append(u_j in self.crm.F)
                info_buffer.append({})

        return (
            np.array(obs_buffer),
            np.array(action_buffer),
            np.array(obs_next_buffer),
            np.array(reward_buffer),
            np.array(done_buffer),
            np.array(info_buffer),
        )

    def _counterfactual_configurations(
        self, props: set[Symbol]
    ) -> dict[int, list[tuple[int, int]]]:
        """Return phase-compatible counters for the observed event."""
        if Symbol.M in props:
            return {0: [(value, value) for value in range(self.mail_count)]}
        if Symbol.E in props:
            return {0: [(self.mail_count, self.mail_count)]}
        if Symbol.C in props:
            return {
                1: [(value, self.mail_count) for value in range(1, self.mail_count + 1)]
            }
        if Symbol.P in props:
            return {2: [(0, value) for value in range(1, self.mail_count + 1)]}
        return {
            0: [(value, value) for value in range(self.mail_count + 1)],
            1: [(value, self.mail_count) for value in range(1, self.mail_count + 1)],
            2: [(0, value) for value in range(1, self.mail_count + 1)],
        }


class TrainingReturnCallback(BaseCallback):
    """Record actual training episode returns from Monitor infos."""

    def __init__(self) -> None:
        """Create the callback."""
        super().__init__(verbose=0)
        self.points: list[dict[str, float | int]] = []
        self._started_at = 0.0

    def _on_training_start(self) -> None:
        self._started_at = time.perf_counter()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            episode = info.get("episode")
            if episode is None:
                continue
            self.points.append(
                {
                    "episode": len(self.points) + 1,
                    "timesteps": int(self.num_timesteps),
                    "time_s": float(time.perf_counter() - self._started_at),
                    "return": float(episode["r"]),
                    "length": int(episode["l"]),
                    "epsilon": float(getattr(self.model, "exploration_rate", 0.0)),
                }
            )
        return True


def make_env(
    *, mail_count: int, max_steps: int, start_row: int, start_col: int, seed: int
):
    """Create one high-mail OfficeWorld cross-product environment."""
    np.random.seed(seed)
    env = OfficeWorldDqnCrossProduct(
        mail_count=mail_count,
        max_steps=max_steps,
        start_row=start_row,
        start_col=start_col,
    )
    env.reset(seed=seed)
    return env


def run_benchmark(
    *,
    algorithms: Iterable[str],
    backends: Iterable[str],
    timesteps: int,
    seed: int,
    mail_count: int,
    max_steps: int,
    start_row: int,
    start_col: int,
    buffer_size: int,
    batch_size: int,
    learning_starts: int,
    device: str,
) -> dict[str, Any]:
    """Run DQN/CDQN on high-mail OfficeWorld."""
    selected_algorithms = _expand(algorithms, ALGORITHMS, "algorithm")
    selected_backends = _expand(backends, BACKENDS, "backend")
    runs = []
    for algorithm in selected_algorithms:
        for backend in selected_backends:
            run_seed = seed + 10_000 * selected_algorithms.index(algorithm)
            run_seed += 1_000 * selected_backends.index(backend)
            runs.append(
                run_one(
                    algorithm=algorithm,
                    backend=backend,
                    timesteps=timesteps,
                    seed=run_seed,
                    mail_count=mail_count,
                    max_steps=max_steps,
                    start_row=start_row,
                    start_col=start_col,
                    buffer_size=buffer_size,
                    batch_size=batch_size,
                    learning_starts=learning_starts,
                    device=device,
                )
            )
    return {
        "config": {
            "algorithms": selected_algorithms,
            "backends": selected_backends,
            "timesteps": timesteps,
            "seed": seed,
            "mail_count": mail_count,
            "max_steps": max_steps,
            "start_row": start_row,
            "start_col": start_col,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "learning_starts": learning_starts,
            "device": device,
        },
        "runs": runs,
        "summary": summarize_runs(runs),
        "note": (
            "Curves are training episode returns collected under the active "
            "epsilon-greedy policy. Mail count is fixed at the configured "
            "high value each episode."
        ),
    }


def run_one(
    *,
    algorithm: str,
    backend: str,
    timesteps: int,
    seed: int,
    mail_count: int,
    max_steps: int,
    start_row: int,
    start_col: int,
    buffer_size: int,
    batch_size: int,
    learning_starts: int,
    device: str,
) -> dict[str, Any]:
    """Run one backend and algorithm."""
    env = make_env(
        mail_count=mail_count,
        max_steps=max_steps,
        start_row=start_row,
        start_col=start_col,
        seed=seed,
    )
    env.action_space.seed(seed)
    algorithm_class = load_algorithm(backend, algorithm)
    model = algorithm_class(
        **algorithm_kwargs(
            algorithm=algorithm,
            backend=backend,
            env=env,
            buffer_size=buffer_size,
            batch_size=batch_size,
            learning_starts=learning_starts,
            seed=seed,
            device=device,
        )
    )
    callback = TrainingReturnCallback()
    started_at = time.perf_counter()
    model.learn(total_timesteps=timesteps, callback=callback, log_interval=1000)
    train_time_s = time.perf_counter() - started_at
    env.close()
    final_window = callback.points[-20:]
    return {
        "algorithm": algorithm,
        "backend": backend,
        "seed": seed,
        "timesteps": timesteps,
        "train_time_s": float(train_time_s),
        "episode_count": len(callback.points),
        "final_mean_training_return": (
            float(np.mean([point["return"] for point in final_window]))
            if final_window
            else None
        ),
        "final_mean_episode_length": (
            float(np.mean([point["length"] for point in final_window]))
            if final_window
            else None
        ),
        "training_curve": callback.points,
    }


def load_algorithm(backend: str, algorithm: str):
    """Load the requested implementation."""
    if backend == "sb3" and algorithm == "dqn":
        from stable_baselines3.dqn import DQN

        return DQN
    if backend == "sb3" and algorithm == "cdqn":
        from pycrm.agents.sb3.dqn import CounterfactualDQN

        return CounterfactualDQN
    if backend == "sbx" and algorithm == "dqn":
        from pycrm.agents.sbx.dqn import DQN

        return DQN
    if backend == "sbx" and algorithm == "cdqn":
        from pycrm.agents.sbx.dqn import CounterfactualDQN

        return CounterfactualDQN
    raise ValueError(f"Unknown backend/algorithm pair: {backend}/{algorithm}")


def algorithm_kwargs(
    *,
    algorithm: str,
    backend: str,
    env,
    buffer_size: int,
    batch_size: int,
    learning_starts: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    """Return constructor kwargs for high-mail OfficeWorld."""
    gradient_steps = 4 if algorithm == "cdqn" else 1
    batch = max(batch_size, 256) if algorithm == "cdqn" else batch_size
    del backend
    return {
        "policy": "MlpPolicy",
        "env": env,
        "verbose": 0,
        "learning_rate": 3e-4,
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "batch_size": batch,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": gradient_steps,
        "exploration_fraction": 0.55,
        "exploration_final_eps": 0.05,
        "target_update_interval": 500,
        "policy_kwargs": {"net_arch": [64, 64]},
        "seed": seed,
        "device": device,
    }


def summarize_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize final training-return windows."""
    return [
        {
            "algorithm": run["algorithm"],
            "backend": run["backend"],
            "train_time_s": run["train_time_s"],
            "episode_count": run["episode_count"],
            "final_mean_training_return": run["final_mean_training_return"],
            "final_mean_episode_length": run["final_mean_episode_length"],
        }
        for run in runs
    ]


def write_plots(result: dict[str, Any], output_dir: Path) -> dict[str, str]:
    """Write high-mail OfficeWorld training-return plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    title = (
        "High-Mail OfficeWorld Training Returns "
        f"(mail={result['config']['mail_count']})"
    )
    paths = {
        "training_returns_by_episode": output_dir / "officeworld_returns_episode.png",
        "training_returns_by_time": output_dir / "officeworld_returns_time.png",
    }
    _plot_training_returns(
        result["runs"],
        paths["training_returns_by_episode"],
        x_key="episode",
        x_label="Training episode",
        title=title,
    )
    _plot_training_returns(
        result["runs"],
        paths["training_returns_by_time"],
        x_key="time_s",
        x_label="Wall-clock training time (s)",
        title=title,
    )
    return {key: str(value) for key, value in paths.items()}


def _plot_training_returns(
    runs: list[dict[str, Any]], path: Path, *, x_key: str, x_label: str, title: str
) -> None:
    import matplotlib.pyplot as plt

    colors = {
        ("sb3", "dqn"): "#4C78A8",
        ("sbx", "dqn"): "#F58518",
        ("sb3", "cdqn"): "#54A24B",
        ("sbx", "cdqn"): "#B279A2",
    }
    labels = {
        ("sb3", "dqn"): "SB3 DQN",
        ("sbx", "dqn"): "SBX/JAX DQN",
        ("sb3", "cdqn"): "SB3 C-DQN",
        ("sbx", "cdqn"): "SBX/JAX C-DQN",
    }
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for run in runs:
        curve = run["training_curve"]
        if not curve:
            continue
        key = (run["backend"], run["algorithm"])
        x_values = np.array([point[x_key] for point in curve], dtype=np.float64)
        returns = np.array([point["return"] for point in curve], dtype=np.float64)
        smooth = _moving_average(returns, window=10)
        x_smooth = x_values[len(x_values) - len(smooth) :]
        ax.plot(x_values, returns, color=colors[key], alpha=0.16, linewidth=0.8)
        ax.plot(
            x_smooth,
            smooth,
            color=colors[key],
            linewidth=2.2,
            label=labels[key],
        )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Episode return under training epsilon")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def _expand(values: Iterable[str], allowed: tuple[str, ...], name: str) -> list[str]:
    selected = list(values)
    if "all" in selected:
        return list(allowed)
    unknown = set(selected) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown {name}s: {sorted(unknown)}")
    return selected


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run DQN/CDQN on high-mail OfficeWorld."
    )
    parser.add_argument("--algorithms", nargs="+", default=["all"])
    parser.add_argument("--backends", nargs="+", default=["sb3", "sbx"])
    parser.add_argument("--timesteps", type=int, default=80_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mail-count", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--start-row", type=int, default=6)
    parser.add_argument("--start-col", type=int, default=9)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("results/jax_officeworld_dqn_convergence/results.json"),
    )
    parser.add_argument(
        "--plot-output-dir",
        type=Path,
        default=Path("results/jax_officeworld_dqn_convergence"),
    )
    return parser.parse_args()


def main() -> None:
    """Run the benchmark CLI."""
    args = parse_args()
    result = run_benchmark(
        algorithms=args.algorithms,
        backends=args.backends,
        timesteps=args.timesteps,
        seed=args.seed,
        mail_count=args.mail_count,
        max_steps=args.max_steps,
        start_row=args.start_row,
        start_col=args.start_col,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        device=args.device,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    plots = write_plots(result, args.plot_output_dir)
    print(f"Wrote JSON to {args.json_output}")
    for name, path in plots.items():
        print(f"Wrote {name} plot to {path}")
    for row in result["summary"]:
        print(
            f"{row['backend']} {row['algorithm']}: "
            f"return={row['final_mean_training_return']:.3f}, "
            f"length={row['final_mean_episode_length']:.2f}, "
            f"time={row['train_time_s']:.2f}s"
        )


if __name__ == "__main__":
    main()
