import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from examples.jax.tabular_q_learning_speed import (
    TASKS,
    TASKS_BY_NAME,
    TabularBenchmarkTask,
)


def run_parallel_env_benchmark(
    *,
    task_names: Sequence[str] | None = None,
    env_counts: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
    steps_per_env: int = 1024,
    repeats: int = 3,
    seed: int = 0,
) -> dict[str, Any]:
    """Benchmark random-policy rollouts over different parallel env counts."""
    if steps_per_env <= 0:
        raise ValueError("steps_per_env must be positive.")
    if repeats <= 0:
        raise ValueError("repeats must be positive.")
    if len(env_counts) == 0:
        raise ValueError("env_counts must be non-empty.")
    if any(env_count <= 0 for env_count in env_counts):
        raise ValueError("env_counts must contain positive integers.")

    tasks = _select_tasks(task_names)
    env_counts = tuple(int(env_count) for env_count in env_counts)
    task_results = {}

    for task_idx, task in enumerate(tasks):
        task_results[task.name] = _run_task_parallel_env_benchmark(
            task=task,
            env_counts=env_counts,
            steps_per_env=steps_per_env,
            repeats=repeats,
            seed=seed + 100_000 * task_idx,
        )

    return {
        "config": {
            "tasks": [task.name for task in tasks],
            "env_counts": list(env_counts),
            "steps_per_env": steps_per_env,
            "repeats": repeats,
            "seed": seed,
        },
        "tasks": task_results,
        "note": (
            "Each timing measures random-policy rollout transitions. "
            "JAX first-call compile time is reported separately from steady-state "
            "runtime."
        ),
    }


def save_plots(result: dict[str, Any], output_dir: Path) -> dict[str, str]:
    """Save parallel-env throughput, time, and speedup plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    throughput_path = output_dir / "parallel_env_throughput.png"
    times_path = output_dir / "parallel_env_times.png"
    speedup_path = output_dir / "parallel_env_speedup.png"

    _plot_parallel_env_throughput(result, throughput_path)
    _plot_parallel_env_times(result, times_path)
    _plot_parallel_env_speedup(result, speedup_path)

    return {
        "throughput": str(throughput_path),
        "times": str(times_path),
        "speedup": str(speedup_path),
    }


def _run_task_parallel_env_benchmark(
    *,
    task: TabularBenchmarkTask,
    env_counts: Sequence[int],
    steps_per_env: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    python_times = []
    python_returns = []
    jax_compile_times = []
    jax_steady_times = []
    jax_returns = []

    for count_idx, env_count in enumerate(env_counts):
        python_count_times, python_count_returns = _time_python_parallel_rollout(
            task=task,
            env_count=env_count,
            steps_per_env=steps_per_env,
            repeats=repeats,
            seed=seed + 10_000 * count_idx,
        )
        rollout_jit = _make_jax_parallel_rollout_fn(
            task=task,
            env_count=env_count,
            steps_per_env=steps_per_env,
        )
        (
            jax_compile_first_call_s,
            jax_count_times,
            jax_count_returns,
        ) = _time_jax_parallel_rollout(
            rollout_jit,
            repeats=repeats,
            seed=seed + 50_000 + 10_000 * count_idx,
        )

        python_times.append(python_count_times)
        python_returns.append(python_count_returns)
        jax_compile_times.append(jax_compile_first_call_s)
        jax_steady_times.append(jax_count_times)
        jax_returns.append(jax_count_returns)

    transitions = np.asarray(env_counts, dtype=np.float64) * float(steps_per_env)
    python_mean_s = np.asarray([np.mean(row) for row in python_times], dtype=np.float64)
    jax_steady_mean_s = np.asarray(
        [np.mean(row) for row in jax_steady_times], dtype=np.float64
    )
    jax_compile_first_call_s = np.asarray(jax_compile_times, dtype=np.float64)

    return {
        "title": task.title,
        "config": {
            "env_counts": list(env_counts),
            "steps_per_env": steps_per_env,
            "repeats": repeats,
            "seed": seed,
        },
        "python": {
            "label": "Normal Python",
            "times_s": [[float(value) for value in row] for row in python_times],
            "mean_s": [float(value) for value in python_mean_s],
            "steps_per_second": [
                float(steps / seconds)
                for steps, seconds in zip(transitions, python_mean_s, strict=True)
            ],
            "total_returns": [
                [float(value) for value in row] for row in python_returns
            ],
        },
        "jax": {
            "label": "JAX vmap",
            "compile_first_call_s": [
                float(value) for value in jax_compile_first_call_s
            ],
            "steady_times_s": [
                [float(value) for value in row] for row in jax_steady_times
            ],
            "steady_mean_s": [float(value) for value in jax_steady_mean_s],
            "steps_per_second": [
                float(steps / seconds)
                for steps, seconds in zip(transitions, jax_steady_mean_s, strict=True)
            ],
            "compile_plus_run_steps_per_second": [
                float(steps / seconds)
                for steps, seconds in zip(
                    transitions,
                    jax_compile_first_call_s,
                    strict=True,
                )
            ],
            "total_returns": [[float(value) for value in row] for row in jax_returns],
        },
        "speedup": {
            "steady_state_ratio": [
                float(python_s / jax_s)
                for python_s, jax_s in zip(
                    python_mean_s,
                    jax_steady_mean_s,
                    strict=True,
                )
            ],
        },
    }


def _time_python_parallel_rollout(
    *,
    task: TabularBenchmarkTask,
    env_count: int,
    steps_per_env: int,
    repeats: int,
    seed: int,
) -> tuple[list[float], list[float]]:
    times = []
    returns = []

    for repeat_idx in range(repeats):
        np.random.seed(seed + repeat_idx)
        rng = np.random.default_rng(seed + repeat_idx)
        envs = [task.make_python_env(steps_per_env) for _ in range(env_count)]

        start = time.perf_counter()
        total_return = 0.0
        for env in envs:
            env.reset()

        for _ in range(steps_per_env):
            actions = rng.integers(task.n_actions, size=env_count)
            for env, action in zip(envs, actions, strict=True):
                _, reward, terminated, truncated, _ = env.step(int(action))
                total_return += float(reward)
                if terminated or truncated:
                    env.reset()

        times.append(time.perf_counter() - start)
        returns.append(total_return)

    return times, returns


def _time_jax_parallel_rollout(
    rollout_jit,
    *,
    repeats: int,
    seed: int,
) -> tuple[float, list[float], list[float]]:
    compile_key = jax.random.PRNGKey(seed)
    start = time.perf_counter()
    _, compile_return = rollout_jit(compile_key)
    compile_return.block_until_ready()
    compile_first_call_s = time.perf_counter() - start

    times = []
    returns = []
    for repeat_idx in range(repeats):
        key = jax.random.PRNGKey(seed + 10_000 + repeat_idx)
        start = time.perf_counter()
        _, total_return = rollout_jit(key)
        total_return.block_until_ready()
        times.append(time.perf_counter() - start)
        returns.append(float(np.asarray(total_return)))

    return float(compile_first_call_s), times, returns


def _make_jax_parallel_rollout_fn(
    *,
    task: TabularBenchmarkTask,
    env_count: int,
    steps_per_env: int,
):
    env = task.make_jax_env(steps_per_env)

    def rollout(key):
        reset_key, scan_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, env_count)
        states, obs = jax.vmap(env.reset)(reset_keys)
        returns = jnp.zeros(env_count, dtype=jnp.float32)
        step_keys = jax.random.split(scan_key, steps_per_env)

        def step_body(carry, step_key):
            states, obs, returns = carry
            action_key, env_key, reset_key = jax.random.split(step_key, 3)
            actions = jax.random.randint(
                action_key,
                (env_count,),
                0,
                task.n_actions,
                dtype=jnp.int32,
            )
            env_keys = jax.random.split(env_key, env_count)
            reset_keys = jax.random.split(reset_key, env_count)
            (
                stepped_states,
                stepped_obs,
                reward,
                terminated,
                truncated,
                valid,
            ) = jax.vmap(env.step)(states, actions, env_keys)
            done = terminated | truncated | (~valid)
            reset_states, reset_obs = jax.vmap(env.reset)(reset_keys)
            states = jax.tree_util.tree_map(
                lambda stepped, reset: _select_reset(done, stepped, reset),
                stepped_states,
                reset_states,
            )
            obs = _select_reset(done, stepped_obs, reset_obs)
            returns = returns + reward
            return (states, obs, returns), jnp.sum(reward)

        (_, _, returns), reward_trace = jax.lax.scan(
            step_body,
            (states, obs, returns),
            step_keys,
        )
        return returns, jnp.sum(reward_trace)

    return jax.jit(rollout)


def _select_reset(done, stepped, reset):
    mask = done
    while mask.ndim < stepped.ndim:
        mask = mask[..., None]
    return jnp.where(mask, reset, stepped)


def _select_tasks(task_names: Sequence[str] | None) -> tuple[TabularBenchmarkTask, ...]:
    if task_names is None or len(task_names) == 0 or "all" in task_names:
        return TASKS

    unknown = sorted(set(task_names) - set(TASKS_BY_NAME))
    if unknown:
        raise ValueError(
            "Unknown task name(s): "
            + ", ".join(unknown)
            + ". Choices are: all, "
            + ", ".join(TASKS_BY_NAME)
            + "."
        )
    return tuple(TASKS_BY_NAME[name] for name in task_names)


def _plot_parallel_env_throughput(result: dict[str, Any], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    task_items = list(result["tasks"].items())
    fig, axes = plt.subplots(
        len(task_items),
        1,
        figsize=(10.5, max(4.0, 3.4 * len(task_items))),
        squeeze=False,
        sharex=True,
    )

    for ax, (_, task_result) in zip(axes[:, 0], task_items, strict=True):
        env_counts = np.asarray(task_result["config"]["env_counts"])
        ax.plot(
            env_counts,
            task_result["python"]["steps_per_second"],
            marker="o",
            linewidth=2.2,
            label="Normal Python",
            color="#2f6fbb",
        )
        ax.plot(
            env_counts,
            task_result["jax"]["steps_per_second"],
            marker="o",
            linewidth=2.2,
            label="JAX vmap steady",
            color="#c95f2d",
        )
        ax.set_title(task_result["title"])
        ax.set_ylabel("Transitions / second")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(env_counts)
        ax.set_xticklabels([str(value) for value in env_counts])
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")

    axes[-1, 0].set_xlabel("Parallel environments")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_parallel_env_times(result: dict[str, Any], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    task_items = list(result["tasks"].items())
    fig, axes = plt.subplots(
        len(task_items),
        1,
        figsize=(10.5, max(4.0, 3.4 * len(task_items))),
        squeeze=False,
        sharex=True,
    )

    for ax, (_, task_result) in zip(axes[:, 0], task_items, strict=True):
        env_counts = np.asarray(task_result["config"]["env_counts"])
        ax.plot(
            env_counts,
            task_result["python"]["mean_s"],
            marker="o",
            linewidth=2.2,
            label="Normal Python",
            color="#2f6fbb",
        )
        ax.plot(
            env_counts,
            task_result["jax"]["steady_mean_s"],
            marker="o",
            linewidth=2.2,
            label="JAX vmap steady",
            color="#c95f2d",
        )
        ax.plot(
            env_counts,
            task_result["jax"]["compile_first_call_s"],
            marker="o",
            linewidth=1.8,
            linestyle="--",
            label="JAX compile+run",
            color="#767676",
        )
        ax.set_title(task_result["title"])
        ax.set_ylabel("Seconds")
        ax.set_xscale("log", base=2)
        ax.set_xticks(env_counts)
        ax.set_xticklabels([str(value) for value in env_counts])
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")

    axes[-1, 0].set_xlabel("Parallel environments")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_parallel_env_speedup(result: dict[str, Any], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    task_items = list(result["tasks"].items())
    fig, axes = plt.subplots(
        len(task_items),
        1,
        figsize=(10.5, max(4.0, 3.4 * len(task_items))),
        squeeze=False,
        sharex=True,
    )

    for ax, (_, task_result) in zip(axes[:, 0], task_items, strict=True):
        env_counts = np.asarray(task_result["config"]["env_counts"])
        ax.plot(
            env_counts,
            task_result["speedup"]["steady_state_ratio"],
            marker="o",
            linewidth=2.2,
            color="#4f7f45",
        )
        ax.axhline(1.0, color="#444444", linewidth=1.2, linestyle="--")
        ax.set_title(task_result["title"])
        ax.set_ylabel("Python / JAX time")
        ax.set_xscale("log", base=2)
        ax.set_xticks(env_counts)
        ax.set_xticklabels([str(value) for value in env_counts])
        ax.grid(True, linestyle="--", alpha=0.35)

    axes[-1, 0].set_xlabel("Parallel environments")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _without_history(result: dict[str, Any]) -> dict[str, Any]:
    stripped = {key: value for key, value in result.items() if key != "tasks"}
    stripped["tasks"] = {}
    for task_name, task_result in result["tasks"].items():
        stripped["tasks"][task_name] = {
            "title": task_result["title"],
            "config": task_result["config"],
            "python": {
                key: value
                for key, value in task_result["python"].items()
                if key != "total_returns"
            },
            "jax": {
                key: value
                for key, value in task_result["jax"].items()
                if key != "total_returns"
            },
            "speedup": task_result["speedup"],
        }
    return stripped


def main() -> None:
    """Run the command-line parallel environment benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark Python loops versus JAX vmap over parallel envs."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["all"],
        choices=["all", *TASKS_BY_NAME.keys()],
        help="Subset of tabular tasks to run.",
    )
    parser.add_argument(
        "--env-counts",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64],
        help="Parallel environment counts to benchmark.",
    )
    parser.add_argument("--steps-per-env", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--plot-output-dir", type=Path, default=None)
    parser.add_argument("--print-history", action="store_true")
    args = parser.parse_args()

    result = run_parallel_env_benchmark(
        task_names=args.tasks,
        env_counts=args.env_counts,
        steps_per_env=args.steps_per_env,
        repeats=args.repeats,
        seed=args.seed,
    )

    if args.plot_output_dir is not None:
        result["plots"] = save_plots(result, args.plot_output_dir)

    text = json.dumps(result, indent=2)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")

    if args.print_history:
        print(text)
    else:
        print(json.dumps(_without_history(result), indent=2))


if __name__ == "__main__":
    main()
