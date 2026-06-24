import argparse
import json
import time
from collections.abc import Iterable
from enum import Enum, auto
from itertools import pairwise
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from pycrm.automaton import RewardMachine
from pycrm.crossproduct import CrossProduct
from pycrm.label import LabellingFunction

ALGORITHMS = ("dqn", "cdqn")
BACKENDS = ("sb3", "sbx")
TARGET_SEQUENCE = (
    "A",
    "B",
    "C",
    "D",
    "A",
    "B",
    "C",
    "D",
    "A",
    "B",
    "C",
    "D",
)
STEP_PENALTY = -0.01
INTERMEDIATE_REWARD = 0.5
TERMINAL_REWARD = 3.0
GRID_SIZE = 9
START_POS = (4, 4)
LANDMARKS = {
    "A": (0, 0),
    "B": (0, GRID_SIZE - 1),
    "C": (GRID_SIZE - 1, GRID_SIZE - 1),
    "D": (GRID_SIZE - 1, 0),
}


def _manhattan(first: tuple[int, int], second: tuple[int, int]) -> int:
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


OPTIMAL_STEPS = _manhattan(START_POS, LANDMARKS[TARGET_SEQUENCE[0]]) + sum(
    _manhattan(LANDMARKS[source], LANDMARKS[target])
    for source, target in pairwise(TARGET_SEQUENCE)
)
OPTIMAL_RETURN = (
    (len(TARGET_SEQUENCE) - 1) * INTERMEDIATE_REWARD
    + TERMINAL_REWARD
    + (OPTIMAL_STEPS - len(TARGET_SEQUENCE)) * STEP_PENALTY
)


class Symbol(Enum):
    """Events in the sequence gridworld."""

    A = auto()
    B = auto()
    C = auto()
    D = auto()


class SequenceGridWorld(gym.Env):
    """Deterministic gridworld with four landmark events."""

    metadata = {"render_modes": []}

    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3

    def __init__(self) -> None:
        """Initialize the ground environment."""
        super().__init__()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=GRID_SIZE - 1,
            shape=(2,),
            dtype=np.int32,
        )
        self.position = np.array(START_POS, dtype=np.int32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset to the middle of the grid."""
        super().reset(seed=seed, options=options)
        self.position = np.array(START_POS, dtype=np.int32)
        return self._obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Move on the grid; rewards come from the reward machine."""
        if int(action) == self.RIGHT:
            self.position[1] = min(self.position[1] + 1, GRID_SIZE - 1)
        elif int(action) == self.LEFT:
            self.position[1] = max(self.position[1] - 1, 0)
        elif int(action) == self.UP:
            self.position[0] = max(self.position[0] - 1, 0)
        elif int(action) == self.DOWN:
            self.position[0] = min(self.position[0] + 1, GRID_SIZE - 1)
        else:
            raise ValueError(f"Invalid action {action}.")
        return self._obs(), 0.0, False, False, {}

    def _obs(self) -> np.ndarray:
        return self.position.copy()


class SequenceLabellingFunction(LabellingFunction[np.ndarray, int]):
    """Label visits to the four landmark cells."""

    @LabellingFunction.event
    def test_a(
        self, obs: np.ndarray, action: int, next_obs: np.ndarray
    ) -> Symbol | None:
        """Emit A when the next position is landmark A."""
        del obs, action
        if tuple(int(value) for value in next_obs) == LANDMARKS["A"]:
            return Symbol.A
        return None

    @LabellingFunction.event
    def test_b(
        self, obs: np.ndarray, action: int, next_obs: np.ndarray
    ) -> Symbol | None:
        """Emit B when the next position is landmark B."""
        del obs, action
        if tuple(int(value) for value in next_obs) == LANDMARKS["B"]:
            return Symbol.B
        return None

    @LabellingFunction.event
    def test_c(
        self, obs: np.ndarray, action: int, next_obs: np.ndarray
    ) -> Symbol | None:
        """Emit C when the next position is landmark C."""
        del obs, action
        if tuple(int(value) for value in next_obs) == LANDMARKS["C"]:
            return Symbol.C
        return None

    @LabellingFunction.event
    def test_d(
        self, obs: np.ndarray, action: int, next_obs: np.ndarray
    ) -> Symbol | None:
        """Emit D when the next position is landmark D."""
        del obs, action
        if tuple(int(value) for value in next_obs) == LANDMARKS["D"]:
            return Symbol.D
        return None


class SequenceRewardMachine(RewardMachine):
    """Reward machine for a long alternating endpoint sequence."""

    def __init__(self) -> None:
        """Initialize the reward machine."""
        super().__init__(env_prop_enum=Symbol)

    @property
    def u_0(self) -> int:
        """Return the initial machine state."""
        return 0

    def _get_state_transition_function(self) -> dict:
        """Return the state transition function."""
        return {
            state: {
                target: -1 if state == len(TARGET_SEQUENCE) - 1 else state + 1,
                f"NOT {target}": state,
            }
            for state, target in enumerate(TARGET_SEQUENCE)
        }

    def _get_reward_transition_function(self) -> dict:
        """Return the reward transition function."""
        return {
            state: {
                target: (
                    TERMINAL_REWARD
                    if state == len(TARGET_SEQUENCE) - 1
                    else INTERMEDIATE_REWARD
                ),
                f"NOT {target}": STEP_PENALTY,
            }
            for state, target in enumerate(TARGET_SEQUENCE)
        }


class SequenceCrossProduct(CrossProduct[np.ndarray, np.ndarray, int, None]):
    """Cross-product MDP with a clean vector observation space."""

    def __init__(self, max_steps: int) -> None:
        """Initialize the sequence task."""
        super().__init__(
            ground_env=SequenceGridWorld(),
            machine=SequenceRewardMachine(),
            lf=SequenceLabellingFunction(),
            max_steps=max_steps,
        )
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(TARGET_SEQUENCE) + 4,),
            dtype=np.float32,
        )
        self.action_space = self.ground_env.action_space

    def _get_obs(
        self, ground_obs: np.ndarray, u: int, c: tuple[int, ...]
    ) -> np.ndarray:
        """Return [normalized row/col, one-hot machine state, counter]."""
        del c
        u_enc = self.crm.encode_machine_state(u).astype(np.float32)
        return np.concatenate(
            (
                ground_obs.astype(np.float32) / float(GRID_SIZE - 1),
                u_enc,
                np.array([0.0], dtype=np.float32),
            )
        )

    def to_ground_obs(self, obs: np.ndarray) -> np.ndarray:
        """Recover the ground row and column from a cross-product observation."""
        row = int(np.rint(float(obs[0]) * (GRID_SIZE - 1)))
        col = int(np.rint(float(obs[1]) * (GRID_SIZE - 1)))
        return np.array(
            [
                np.clip(row, 0, GRID_SIZE - 1),
                np.clip(col, 0, GRID_SIZE - 1),
            ],
            dtype=np.int32,
        )


class TrainingReturnCallback(BaseCallback):
    """Record actual rollout episode returns under the training policy."""

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


def make_env(max_steps: int, seed: int):
    """Create one sequence cross-product environment."""
    env = SequenceCrossProduct(max_steps=max_steps)
    env.reset(seed=seed)
    return env


def run_benchmark(
    *,
    algorithms: Iterable[str],
    backends: Iterable[str],
    timesteps: int,
    seed: int,
    max_steps: int,
    buffer_size: int,
    batch_size: int,
    learning_starts: int,
    device: str,
) -> dict[str, Any]:
    """Run deep DQN/CDQN convergence checks on the sequence gridworld."""
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
                    max_steps=max_steps,
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
            "max_steps": max_steps,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "learning_starts": learning_starts,
            "device": device,
        },
        "runs": runs,
        "summary": summarize_runs(runs),
        "note": (
            "Curves are training episode returns collected under the active "
            "epsilon-greedy policy, not separate greedy evaluation rollouts."
        ),
    }


def run_one(
    *,
    algorithm: str,
    backend: str,
    timesteps: int,
    seed: int,
    max_steps: int,
    buffer_size: int,
    batch_size: int,
    learning_starts: int,
    device: str,
) -> dict[str, Any]:
    """Run one backend and algorithm."""
    np.random.seed(seed)
    env = make_env(max_steps, seed)
    env.action_space.seed(seed)
    algorithm_class = load_algorithm(backend, algorithm)
    kwargs = algorithm_kwargs(
        algorithm=algorithm,
        backend=backend,
        env=env,
        buffer_size=buffer_size,
        batch_size=batch_size,
        learning_starts=learning_starts,
        seed=seed,
        device=device,
    )
    model = algorithm_class(**kwargs)
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
    """Return tuned constructor kwargs for the small convergence task."""
    del algorithm, backend
    return {
        "policy": "MlpPolicy",
        "env": env,
        "verbose": 0,
        "learning_rate": 1e-3,
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "batch_size": batch_size,
        "gamma": 0.95,
        "train_freq": 1,
        "gradient_steps": 1,
        "exploration_fraction": 0.35,
        "exploration_final_eps": 0.05,
        "target_update_interval": 100,
        "policy_kwargs": {"net_arch": [32, 32]},
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
    """Write training-return plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "training_returns_by_episode": output_dir / "gridworld_returns_episode.png",
        "training_returns_by_time": output_dir / "gridworld_returns_time.png",
    }
    _plot_training_returns(
        result["runs"],
        paths["training_returns_by_episode"],
        x_key="episode",
        x_label="Training episode",
    )
    _plot_training_returns(
        result["runs"],
        paths["training_returns_by_time"],
        x_key="time_s",
        x_label="Wall-clock training time (s)",
    )
    return {key: str(value) for key, value in paths.items()}


def _plot_training_returns(
    runs: list[dict[str, Any]], path: Path, *, x_key: str, x_label: str
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
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for run in runs:
        curve = run["training_curve"]
        if not curve:
            continue
        key = (run["backend"], run["algorithm"])
        x_values = np.array([point[x_key] for point in curve], dtype=np.float64)
        returns = np.array([point["return"] for point in curve], dtype=np.float64)
        smooth = _moving_average(returns, window=20)
        x_smooth = x_values[len(x_values) - len(smooth) :]
        ax.plot(x_values, returns, color=colors[key], alpha=0.18, linewidth=0.8)
        ax.plot(
            x_smooth,
            smooth,
            color=colors[key],
            linewidth=2.2,
            label=labels[key],
        )
    ax.axhline(
        OPTIMAL_RETURN,
        color="#222222",
        linewidth=1.0,
        alpha=0.6,
        linestyle="--",
    )
    ax.set_title("Extended Sequence Gridworld Training Returns")
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
        description="Run deep DQN/CDQN convergence on a small sequence gridworld."
    )
    parser.add_argument("--algorithms", nargs="+", default=["all"])
    parser.add_argument("--backends", nargs="+", default=["sb3", "sbx"])
    parser.add_argument("--timesteps", type=int, default=40_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-starts", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("results/jax_gridworld_dqn_convergence/results.json"),
    )
    parser.add_argument(
        "--plot-output-dir",
        type=Path,
        default=Path("results/jax_gridworld_dqn_convergence"),
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
        max_steps=args.max_steps,
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
