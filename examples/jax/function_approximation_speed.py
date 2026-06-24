import argparse
import importlib
import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

TASKS: dict[str, dict[str, Any]] = {
    "rm-discrete": {
        "title": "Reward Machine Discrete PuckWorld",
        "core_module": "examples.rm.discrete.core",
        "machine": "PuckWorldRewardMachine",
        "control": "discrete",
        "max_steps": 1000,
        "example_timesteps": 250_000,
    },
    "crm-discrete": {
        "title": "Counting Reward Machine Discrete PuckWorld",
        "core_module": "examples.crm.discrete.core",
        "machine": "PuckWorldCountingRewardMachine",
        "control": "discrete",
        "max_steps": 1000,
        "example_timesteps": 1_000_000,
    },
    "rm-continuous": {
        "title": "Reward Machine Continuous PuckWorld",
        "core_module": "examples.rm.continuous.core",
        "machine": "PuckWorldRewardMachine",
        "control": "continuous",
        "max_steps": 500,
        "example_timesteps": 250_000,
    },
    "crm-continuous": {
        "title": "Counting Reward Machine Continuous PuckWorld",
        "core_module": "examples.crm.continuous.core",
        "machine": "PuckWorldCountingRewardMachine",
        "control": "continuous",
        "max_steps": 100,
        "example_timesteps": 50_000,
    },
}

DISCRETE_ALGORITHMS = ("dqn", "cdqn")
CONTINUOUS_ALGORITHMS = ("ddpg", "cddpg", "sac", "csac", "td3", "ctd3")
BACKENDS = ("sb3", "sbx")


class EvaluationCallback(BaseCallback):
    """Record deterministic evaluation returns against elapsed wall time."""

    def __init__(
        self,
        *,
        make_env,
        eval_freq: int,
        eval_episodes: int,
        seed: int,
        verbose: int = 0,
    ) -> None:
        """Create the callback."""
        super().__init__(verbose=verbose)
        self.make_env = make_env
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.seed = seed
        self.points: list[dict[str, float | int]] = []
        self._started_at = 0.0
        self._last_eval_at = 0

    def _on_training_start(self) -> None:
        self._started_at = time.perf_counter()
        self._last_eval_at = 0

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.eval_episodes <= 0:
            return True
        if self.num_timesteps - self._last_eval_at < self.eval_freq:
            return True

        self._last_eval_at = self.num_timesteps
        mean_return, std_return = evaluate_policy(
            self.model,
            self.make_env,
            n_episodes=self.eval_episodes,
            seed=self.seed + self.num_timesteps,
        )
        self.points.append(
            {
                "timesteps": int(self.num_timesteps),
                "time_s": float(time.perf_counter() - self._started_at),
                "mean_return": float(mean_return),
                "std_return": float(std_return),
            }
        )
        return True


def run_benchmark(
    *,
    tasks: Iterable[str],
    algorithms: Iterable[str],
    backends: Iterable[str],
    timesteps: int,
    repeats: int,
    seed: int,
    max_steps: int | None,
    eval_freq: int,
    eval_episodes: int,
    buffer_size: int,
    batch_size: int,
    learning_starts: int,
    device: str,
    original_example_timesteps: bool,
) -> dict[str, Any]:
    """Run the selected function-approximation benchmark matrix."""
    selected_tasks = _expand_tasks(tasks)
    selected_backends = _expand_backends(backends)
    results = []

    for task_index, task_name in enumerate(selected_tasks):
        task = TASKS[task_name]
        task_algorithms = _expand_algorithms(algorithms, task["control"])
        task_timesteps = (
            int(task["example_timesteps"]) if original_example_timesteps else timesteps
        )
        task_max_steps = int(task["max_steps"] if max_steps is None else max_steps)

        for algorithm in task_algorithms:
            for backend in selected_backends:
                for repeat in range(repeats):
                    run_seed = seed + 10_000 * task_index + 100 * repeat
                    results.append(
                        run_one(
                            task_name=task_name,
                            algorithm=algorithm,
                            backend=backend,
                            timesteps=task_timesteps,
                            max_steps=task_max_steps,
                            eval_freq=eval_freq,
                            eval_episodes=eval_episodes,
                            buffer_size=buffer_size,
                            batch_size=batch_size,
                            learning_starts=learning_starts,
                            device=device,
                            seed=run_seed,
                            repeat=repeat,
                        )
                    )

    return {
        "config": {
            "tasks": selected_tasks,
            "algorithms": list(algorithms),
            "backends": selected_backends,
            "timesteps": timesteps,
            "repeats": repeats,
            "seed": seed,
            "max_steps_override": max_steps,
            "eval_freq": eval_freq,
            "eval_episodes": eval_episodes,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "learning_starts": learning_starts,
            "device": device,
            "original_example_timesteps": original_example_timesteps,
        },
        "summary": summarize_results(results),
        "runs": results,
        "note": (
            "SBX timings use JAX-backed neural-network updates through SBX, "
            "but these PuckWorld environments are still Gymnasium/Python envs."
        ),
    }


def run_one(
    *,
    task_name: str,
    algorithm: str,
    backend: str,
    timesteps: int,
    max_steps: int,
    eval_freq: int,
    eval_episodes: int,
    buffer_size: int,
    batch_size: int,
    learning_starts: int,
    device: str,
    seed: int,
    repeat: int,
) -> dict[str, Any]:
    """Run one task, algorithm, backend, and repeat."""
    np.random.seed(seed)

    def make_env(env_seed: int):
        return make_task_env(task_name, max_steps, env_seed)

    env = make_env(seed)
    algorithm_class = load_algorithm(backend, algorithm)
    kwargs = algorithm_kwargs(
        task_name=task_name,
        algorithm=algorithm,
        backend=backend,
        env=env,
        buffer_size=buffer_size,
        batch_size=batch_size,
        learning_starts=learning_starts,
        device=device,
        seed=seed,
    )
    model = algorithm_class(**kwargs)
    callback = EvaluationCallback(
        make_env=make_env,
        eval_freq=eval_freq,
        eval_episodes=eval_episodes,
        seed=seed + 1_000_000,
    )

    started_at = time.perf_counter()
    model.learn(total_timesteps=timesteps, callback=callback, log_interval=10_000)
    train_time_s = time.perf_counter() - started_at

    if not callback.points and eval_episodes > 0:
        mean_return, std_return = evaluate_policy(
            model,
            make_env,
            n_episodes=eval_episodes,
            seed=seed + 2_000_000,
        )
        callback.points.append(
            {
                "timesteps": int(timesteps),
                "time_s": float(train_time_s),
                "mean_return": float(mean_return),
                "std_return": float(std_return),
            }
        )

    final_return = (
        float(callback.points[-1]["mean_return"]) if callback.points else None
    )
    env.close()
    return {
        "task": task_name,
        "task_title": TASKS[task_name]["title"],
        "algorithm": algorithm,
        "backend": backend,
        "repeat": repeat,
        "seed": seed,
        "timesteps": timesteps,
        "max_steps": max_steps,
        "train_time_s": float(train_time_s),
        "steps_per_second": float(timesteps / train_time_s),
        "final_eval_return": final_return,
        "returns_curve": callback.points,
    }


def make_task_env(task_name: str, max_steps: int, seed: int):
    """Create one PuckWorld cross-product environment."""
    task = TASKS[task_name]
    core = importlib.import_module(task["core_module"])
    np.random.seed(seed)
    ground_env = core.PuckWorld()
    labelling_function = core.PuckWorldLabellingFunction()
    machine = getattr(core, task["machine"])()
    env = core.PuckWorldCrossProduct(
        ground_env=ground_env,
        machine=machine,
        lf=labelling_function,
        max_steps=max_steps,
    )
    env.reset(seed=seed)
    return env


def load_algorithm(backend: str, algorithm: str):
    """Load an algorithm implementation for the requested backend."""
    if backend == "sb3":
        return load_sb3_algorithm(algorithm)
    if backend == "sbx":
        return load_sbx_algorithm(algorithm)
    raise ValueError(f"Unknown backend: {backend}")


def load_sb3_algorithm(algorithm: str):
    """Load an SB3/PyTorch algorithm class."""
    if algorithm == "dqn":
        from stable_baselines3.dqn import DQN

        return DQN
    if algorithm == "cdqn":
        from pycrm.agents.sb3.dqn import CounterfactualDQN

        return CounterfactualDQN
    if algorithm == "ddpg":
        from stable_baselines3.ddpg import DDPG

        return DDPG
    if algorithm == "cddpg":
        from pycrm.agents.sb3.ddpg import CounterfactualDDPG

        return CounterfactualDDPG
    if algorithm == "sac":
        from stable_baselines3.sac import SAC

        return SAC
    if algorithm == "csac":
        from pycrm.agents.sb3.sac import CounterfactualSAC

        return CounterfactualSAC
    if algorithm == "td3":
        from stable_baselines3.td3 import TD3

        return TD3
    if algorithm == "ctd3":
        from pycrm.agents.sb3.td3 import CounterfactualTD3

        return CounterfactualTD3
    raise ValueError(f"Unknown algorithm: {algorithm}")


def load_sbx_algorithm(algorithm: str):
    """Load an SBX/JAX algorithm class."""
    if algorithm == "dqn":
        from pycrm.agents.sbx.dqn import DQN

        return DQN
    if algorithm == "cdqn":
        from pycrm.agents.sbx.dqn import CounterfactualDQN

        return CounterfactualDQN
    if algorithm == "ddpg":
        from sbx import DDPG

        return DDPG
    if algorithm == "cddpg":
        from pycrm.agents.sbx.ddpg import CounterfactualDDPG

        return CounterfactualDDPG
    if algorithm == "sac":
        from sbx import SAC

        return SAC
    if algorithm == "csac":
        from pycrm.agents.sbx.sac import CounterfactualSAC

        return CounterfactualSAC
    if algorithm == "td3":
        from sbx import TD3

        return TD3
    if algorithm == "ctd3":
        from pycrm.agents.sbx.td3 import CounterfactualTD3

        return CounterfactualTD3
    raise ValueError(f"Unknown algorithm: {algorithm}")


def algorithm_kwargs(
    *,
    task_name: str,
    algorithm: str,
    backend: str,
    env,
    buffer_size: int,
    batch_size: int,
    learning_starts: int,
    device: str,
    seed: int,
) -> dict[str, Any]:
    """Build algorithm constructor kwargs shared by SB3 and SBX."""
    kwargs: dict[str, Any] = {
        "policy": "MlpPolicy",
        "env": env,
        "verbose": 0,
        "buffer_size": buffer_size,
        "batch_size": batch_size,
        "learning_starts": learning_starts,
        "device": device,
        "seed": seed,
    }
    if algorithm in {"dqn", "cdqn"}:
        kwargs["exploration_final_eps"] = 0.1
        if task_name.startswith("crm-"):
            kwargs["exploration_fraction"] = 0.5
        elif algorithm == "cdqn":
            kwargs["exploration_fraction"] = 0.2
        else:
            kwargs["exploration_fraction"] = 0.1
    return kwargs


def evaluate_policy(
    model, make_env, *, n_episodes: int, seed: int
) -> tuple[float, float]:
    """Evaluate a model with deterministic actions."""
    returns = []
    for episode in range(n_episodes):
        np.random.seed(seed + episode)
        env = make_env(seed + episode)
        obs, _ = env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        total_return = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_return += float(reward)
        env.close()
        returns.append(total_return)
    return float(np.mean(returns)), float(np.std(returns))


def summarize_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate repeated runs into mean timing and return rows."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for run in results:
        key = (run["task"], run["algorithm"], run["backend"])
        groups.setdefault(key, []).append(run)

    rows = []
    for (task, algorithm, backend), runs in sorted(groups.items()):
        times = np.array([run["train_time_s"] for run in runs], dtype=np.float64)
        returns = [
            run["final_eval_return"]
            for run in runs
            if run["final_eval_return"] is not None
        ]
        rows.append(
            {
                "task": task,
                "algorithm": algorithm,
                "backend": backend,
                "mean_train_time_s": float(np.mean(times)),
                "std_train_time_s": float(np.std(times)),
                "mean_steps_per_second": float(
                    np.mean([r["steps_per_second"] for r in runs])
                ),
                "mean_final_eval_return": (
                    float(np.mean(returns)) if returns else None
                ),
                "repeats": len(runs),
            }
        )

    sb3_times = {
        (row["task"], row["algorithm"]): row["mean_train_time_s"]
        for row in rows
        if row["backend"] == "sb3"
    }
    for row in rows:
        if row["backend"] == "sbx":
            key = (row["task"], row["algorithm"])
            if key in sb3_times:
                row["speedup_vs_sb3"] = float(
                    sb3_times[key] / row["mean_train_time_s"]
                )
    return rows


def write_plots(results: dict[str, Any], output_dir: Path) -> list[str]:
    """Write return, train-time, and speedup plots."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = []

    returns_steps_path = output_dir / "function_approx_returns_timesteps.png"
    plot_returns_by_timesteps(results["runs"], returns_steps_path)
    plot_paths.append(str(returns_steps_path))

    returns_path = output_dir / "function_approx_returns_time.png"
    plot_returns_by_time(results["runs"], returns_path)
    plot_paths.append(str(returns_path))

    times_path = output_dir / "function_approx_train_times.png"
    plot_train_times(results["summary"], times_path)
    plot_paths.append(str(times_path))

    speedups_path = output_dir / "function_approx_sbx_speedups.png"
    plot_speedups(results["summary"], speedups_path)
    plot_paths.append(str(speedups_path))

    plt.close("all")
    return plot_paths


def plot_returns_by_timesteps(runs: list[dict[str, Any]], path: Path) -> None:
    """Plot evaluation returns against environment timesteps."""
    _plot_returns(runs, path, x_key="timesteps", x_label="Training timesteps")


def plot_returns_by_time(runs: list[dict[str, Any]], path: Path) -> None:
    """Plot evaluation returns against elapsed training time."""
    _plot_returns(runs, path, x_key="time_s", x_label="Wall-clock training time (s)")


def _plot_returns(
    runs: list[dict[str, Any]], path: Path, *, x_key: str, x_label: str
) -> None:
    """Plot evaluation returns against one curve x-axis."""
    import matplotlib.pyplot as plt

    task_names = sorted({run["task"] for run in runs})
    ncols = min(2, len(task_names))
    nrows = int(np.ceil(len(task_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows))
    axes_array = np.atleast_1d(axes).flatten()

    colors = {"sb3": "#4C78A8", "sbx": "#F58518"}
    linestyles = {
        "dqn": "-",
        "cdqn": "--",
        "ddpg": "-",
        "cddpg": "--",
        "sac": ":",
        "csac": "-.",
        "td3": (0, (3, 1, 1, 1)),
        "ctd3": (0, (5, 2)),
    }

    for ax, task in zip(axes_array, task_names, strict=False):
        task_runs = [run for run in runs if run["task"] == task]
        for run in task_runs:
            curve = run["returns_curve"]
            if not curve:
                continue
            label = f"{run['backend'].upper()} {run['algorithm'].upper()}"
            ax.plot(
                [point[x_key] for point in curve],
                [point["mean_return"] for point in curve],
                label=label,
                color=colors[run["backend"]],
                linestyle=linestyles.get(run["algorithm"], "-"),
                alpha=0.85,
            )
        ax.set_title(TASKS[task]["title"])
        ax.set_xlabel(x_label)
        ax.set_ylabel("Mean evaluation return")
        ax.grid(alpha=0.25)
        ax.legend(fontsize="small", ncols=2)

    for ax in axes_array[len(task_names) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=160)


def plot_train_times(summary: list[dict[str, Any]], path: Path) -> None:
    """Plot mean train times for each selected run group."""
    import matplotlib.pyplot as plt

    labels = [
        f"{row['task']}\n{row['algorithm'].upper()} {row['backend'].upper()}"
        for row in summary
    ]
    values = [row["mean_train_time_s"] for row in summary]
    colors = ["#4C78A8" if row["backend"] == "sb3" else "#F58518" for row in summary]

    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(labels)), 5))
    ax.bar(range(len(labels)), values, color=colors)
    ax.set_ylabel("Mean train time (s)")
    ax.set_xticks(range(len(labels)), labels, rotation=65, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def plot_speedups(summary: list[dict[str, Any]], path: Path) -> None:
    """Plot SBX speedups relative to matching SB3 runs."""
    import matplotlib.pyplot as plt

    rows = [row for row in summary if "speedup_vs_sb3" in row]
    labels = [f"{row['task']}\n{row['algorithm'].upper()}" for row in rows]
    values = [row["speedup_vs_sb3"] for row in rows]

    fig, ax = plt.subplots(figsize=(max(8, 0.65 * len(labels)), 5))
    ax.axhline(1.0, color="#222222", linewidth=1.0)
    ax.bar(range(len(labels)), values, color="#F58518")
    ax.set_ylabel("SB3 time / SBX time")
    ax.set_xticks(range(len(labels)), labels, rotation=65, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def _expand_tasks(tasks: Iterable[str]) -> list[str]:
    task_names = list(tasks)
    if "all" in task_names:
        return list(TASKS.keys())
    unknown = set(task_names) - set(TASKS)
    if unknown:
        raise ValueError(f"Unknown tasks: {sorted(unknown)}")
    return task_names


def _expand_backends(backends: Iterable[str]) -> list[str]:
    backend_names = list(backends)
    if "all" in backend_names:
        return list(BACKENDS)
    unknown = set(backend_names) - set(BACKENDS)
    if unknown:
        raise ValueError(f"Unknown backends: {sorted(unknown)}")
    return backend_names


def _expand_algorithms(algorithms: Iterable[str], control: str) -> list[str]:
    algorithm_names = list(algorithms)
    allowed = DISCRETE_ALGORITHMS if control == "discrete" else CONTINUOUS_ALGORITHMS
    if "all" in algorithm_names:
        return list(allowed)
    unknown = set(algorithm_names) - set(allowed)
    if unknown:
        raise ValueError(
            f"Algorithms {sorted(unknown)} are invalid for {control} tasks."
        )
    return algorithm_names


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark PyCRM function-approximation examples with SB3 and SBX."
        )
    )
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--algorithms", nargs="+", default=["all"])
    parser.add_argument("--backends", nargs="+", default=["sb3", "sbx"])
    parser.add_argument("--timesteps", type=int, default=5_000)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-freq", type=int, default=1_000)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-starts", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--original-example-timesteps", action="store_true")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("results/jax_function_approximation/benchmark.json"),
    )
    parser.add_argument(
        "--plot-output-dir",
        type=Path,
        default=Path("results/jax_function_approximation"),
    )
    return parser.parse_args()


def main() -> None:
    """Run the benchmark CLI."""
    args = parse_args()
    results = run_benchmark(
        tasks=args.tasks,
        algorithms=args.algorithms,
        backends=args.backends,
        timesteps=args.timesteps,
        repeats=args.repeats,
        seed=args.seed,
        max_steps=args.max_steps,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        device=args.device,
        original_example_timesteps=args.original_example_timesteps,
    )

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot_paths = write_plots(results, args.plot_output_dir)

    print(f"Wrote benchmark JSON to {args.json_output}")
    for path in plot_paths:
        print(f"Wrote plot to {path}")
    for row in results["summary"]:
        speedup = row.get("speedup_vs_sb3")
        suffix = "" if speedup is None else f", speedup vs SB3={speedup:.2f}x"
        print(
            f"{row['task']} {row['algorithm']} {row['backend']}: "
            f"{row['mean_train_time_s']:.3f}s{suffix}"
        )


if __name__ == "__main__":
    main()
