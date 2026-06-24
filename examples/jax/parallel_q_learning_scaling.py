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


def run_parallel_learning_benchmark(
    *,
    task_names: Sequence[str] | None = None,
    env_counts: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
    chunks: int = 20,
    updates_per_chunk: int = 64,
    max_steps: int = 200,
    repeats: int = 3,
    seed: int = 0,
    counter_cap: int | None = None,
    learning_rate: float = 0.1,
    discount_factor: float = 0.99,
    epsilon: float = 0.1,
    eval_episodes: int = 32,
    eval_max_steps: int | None = None,
) -> dict[str, Any]:
    """Benchmark how parallel JAX envs change Q-learning wall-clock curves."""
    if chunks <= 0:
        raise ValueError("chunks must be positive.")
    if updates_per_chunk <= 0:
        raise ValueError("updates_per_chunk must be positive.")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    if repeats <= 0:
        raise ValueError("repeats must be positive.")
    if eval_episodes <= 0:
        raise ValueError("eval_episodes must be positive.")
    if len(env_counts) == 0:
        raise ValueError("env_counts must be non-empty.")
    if any(env_count <= 0 for env_count in env_counts):
        raise ValueError("env_counts must contain positive integers.")
    if counter_cap is not None and counter_cap < 0:
        raise ValueError("counter_cap must be non-negative.")

    eval_max_steps = max_steps if eval_max_steps is None else eval_max_steps
    tasks = _select_tasks(task_names)
    env_counts = tuple(int(env_count) for env_count in env_counts)
    task_results = {}

    for task_idx, task in enumerate(tasks):
        task_counter_cap = (
            task.default_counter_cap if counter_cap is None else counter_cap
        )
        task_results[task.name] = _run_task_parallel_learning_benchmark(
            task=task,
            env_counts=env_counts,
            chunks=chunks,
            updates_per_chunk=updates_per_chunk,
            max_steps=max_steps,
            repeats=repeats,
            seed=seed + 100_000 * task_idx,
            counter_cap=task_counter_cap,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            eval_episodes=eval_episodes,
            eval_max_steps=eval_max_steps,
        )

    return {
        "config": {
            "tasks": [task.name for task in tasks],
            "env_counts": list(env_counts),
            "chunks": chunks,
            "updates_per_chunk": updates_per_chunk,
            "max_steps": max_steps,
            "repeats": repeats,
            "seed": seed,
            "counter_cap": counter_cap,
            "learning_rate": learning_rate,
            "discount_factor": discount_factor,
            "epsilon": epsilon,
            "eval_episodes": eval_episodes,
            "eval_max_steps": eval_max_steps,
        },
        "tasks": task_results,
        "note": (
            "Curves use actual steady-state JAX training time per chunk; "
            "greedy-policy evaluation time and first-call compilation are "
            "reported separately."
        ),
    }


def save_plots(result: dict[str, Any], output_dir: Path) -> dict[str, str]:
    """Save wall-clock, sample, and time-to-target learning plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    wallclock_path = output_dir / "parallel_learning_wallclock.png"
    samples_path = output_dir / "parallel_learning_samples.png"
    target_path = output_dir / "parallel_learning_time_to_target.png"

    _plot_learning_curves(result, wallclock_path, x_key="wall_time_s")
    _plot_learning_curves(result, samples_path, x_key="environment_transitions")
    _plot_time_to_target(result, target_path)

    return {
        "wallclock_returns": str(wallclock_path),
        "sample_returns": str(samples_path),
        "time_to_target": str(target_path),
    }


def _run_task_parallel_learning_benchmark(
    *,
    task: TabularBenchmarkTask,
    env_counts: Sequence[int],
    chunks: int,
    updates_per_chunk: int,
    max_steps: int,
    repeats: int,
    seed: int,
    counter_cap: int,
    learning_rate: float,
    discount_factor: float,
    epsilon: float,
    eval_episodes: int,
    eval_max_steps: int,
) -> dict[str, Any]:
    env_results = {}

    for count_idx, env_count in enumerate(env_counts):
        env_results[str(env_count)] = _run_env_count_learning_benchmark(
            task=task,
            env_count=env_count,
            chunks=chunks,
            updates_per_chunk=updates_per_chunk,
            max_steps=max_steps,
            repeats=repeats,
            seed=seed + 10_000 * count_idx,
            counter_cap=counter_cap,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            eval_episodes=eval_episodes,
            eval_max_steps=eval_max_steps,
        )

    target_return = _infer_target_return(env_results)
    for env_result in env_results.values():
        env_result["time_to_target_s"] = _time_to_target(
            env_result["mean_wall_time_s"],
            env_result["mean_eval_return"],
            target_return,
        )

    return {
        "title": task.title,
        "config": {
            "env_counts": list(env_counts),
            "chunks": chunks,
            "updates_per_chunk": updates_per_chunk,
            "max_steps": max_steps,
            "repeats": repeats,
            "seed": seed,
            "counter_cap": counter_cap,
            "learning_rate": learning_rate,
            "discount_factor": discount_factor,
            "epsilon": epsilon,
            "eval_episodes": eval_episodes,
            "eval_max_steps": eval_max_steps,
            "target_return": target_return,
        },
        "env_counts": env_results,
    }


def _run_env_count_learning_benchmark(
    *,
    task: TabularBenchmarkTask,
    env_count: int,
    chunks: int,
    updates_per_chunk: int,
    max_steps: int,
    repeats: int,
    seed: int,
    counter_cap: int,
    learning_rate: float,
    discount_factor: float,
    epsilon: float,
    eval_episodes: int,
    eval_max_steps: int,
) -> dict[str, Any]:
    init_jit = _make_init_fn(
        task=task,
        env_count=env_count,
        max_steps=max_steps,
        counter_cap=counter_cap,
    )
    train_chunk_jit = _make_train_chunk_fn(
        task=task,
        env_count=env_count,
        updates_per_chunk=updates_per_chunk,
        max_steps=max_steps,
        counter_cap=counter_cap,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
    )
    evaluate_jit = _make_evaluate_fn(
        task=task,
        eval_episodes=eval_episodes,
        eval_max_steps=eval_max_steps,
        max_steps=max_steps,
        counter_cap=counter_cap,
    )

    compile_first_call_s = _compile_learning_functions(
        init_jit=init_jit,
        train_chunk_jit=train_chunk_jit,
        evaluate_jit=evaluate_jit,
        seed=seed,
    )

    repeat_wall_times = []
    repeat_transitions = []
    repeat_returns = []
    repeat_train_times = []

    for repeat_idx in range(repeats):
        key = jax.random.PRNGKey(seed + 100_000 + repeat_idx)
        init_key, eval_key, loop_key = jax.random.split(key, 3)
        q_table, states, obs = init_jit(init_key)
        _block_until_ready((q_table, states, obs))

        elapsed_s = 0.0
        transitions = 0
        wall_times = [0.0]
        transition_counts = [0]
        returns = [float(np.asarray(evaluate_jit(q_table, eval_key)))]
        train_times = []

        for _ in range(chunks):
            loop_key, train_key, eval_key = jax.random.split(loop_key, 3)
            start = time.perf_counter()
            q_table, states, obs = train_chunk_jit(q_table, states, obs, train_key)
            _block_until_ready((q_table, states, obs))
            chunk_time_s = time.perf_counter() - start

            elapsed_s += chunk_time_s
            transitions += env_count * updates_per_chunk
            eval_return = evaluate_jit(q_table, eval_key)
            eval_return.block_until_ready()

            wall_times.append(elapsed_s)
            transition_counts.append(transitions)
            returns.append(float(np.asarray(eval_return)))
            train_times.append(chunk_time_s)

        repeat_wall_times.append(wall_times)
        repeat_transitions.append(transition_counts)
        repeat_returns.append(returns)
        repeat_train_times.append(train_times)

    mean_wall_time_s = np.mean(np.asarray(repeat_wall_times), axis=0)
    mean_transitions = np.mean(np.asarray(repeat_transitions), axis=0)
    mean_eval_return = np.mean(np.asarray(repeat_returns), axis=0)

    return {
        "compile_first_call_s": float(compile_first_call_s),
        "wall_time_s": repeat_wall_times,
        "environment_transitions": repeat_transitions,
        "eval_return": repeat_returns,
        "chunk_train_times_s": repeat_train_times,
        "mean_wall_time_s": [float(value) for value in mean_wall_time_s],
        "mean_environment_transitions": [
            float(value) for value in mean_transitions
        ],
        "mean_eval_return": [float(value) for value in mean_eval_return],
        "final_mean_return": float(mean_eval_return[-1]),
        "total_mean_train_time_s": float(mean_wall_time_s[-1]),
        "transitions_per_second": float(mean_transitions[-1] / mean_wall_time_s[-1]),
    }


def _compile_learning_functions(
    *,
    init_jit,
    train_chunk_jit,
    evaluate_jit,
    seed: int,
) -> float:
    key = jax.random.PRNGKey(seed)
    init_key, train_key, eval_key = jax.random.split(key, 3)
    start = time.perf_counter()
    q_table, states, obs = init_jit(init_key)
    q_table, states, obs = train_chunk_jit(q_table, states, obs, train_key)
    eval_return = evaluate_jit(q_table, eval_key)
    _block_until_ready((q_table, states, obs, eval_return))
    return time.perf_counter() - start


def _make_init_fn(
    *,
    task: TabularBenchmarkTask,
    env_count: int,
    max_steps: int,
    counter_cap: int,
):
    env = task.make_jax_env(max_steps)
    q_shape = task.q_shape(counter_cap)

    def init(key):
        reset_keys = jax.random.split(key, env_count)
        states, obs = jax.vmap(env.reset)(reset_keys)
        q_table = jnp.zeros(q_shape, dtype=jnp.float32)
        return q_table, states, obs

    return jax.jit(init)


def _make_train_chunk_fn(
    *,
    task: TabularBenchmarkTask,
    env_count: int,
    updates_per_chunk: int,
    max_steps: int,
    counter_cap: int,
    learning_rate: float,
    discount_factor: float,
    epsilon: float,
):
    env = task.make_jax_env(max_steps)

    def train_chunk(q_table, states, obs, key):
        step_keys = jax.random.split(key, updates_per_chunk)

        def step_body(carry, step_key):
            q_table, states, obs = carry
            eps_key, action_key, env_key, reset_key = jax.random.split(step_key, 4)

            state_idx = task.jax_obs_batch_index(obs, counter_cap)
            q_values = q_table[state_idx]
            explore = jax.random.uniform(eps_key, (env_count,)) < epsilon
            unvisited = jnp.all(q_values == 0, axis=-1)
            random_actions = jax.random.randint(
                action_key,
                (env_count,),
                0,
                task.n_actions,
                dtype=jnp.int32,
            )
            greedy_actions = jnp.argmax(q_values, axis=-1).astype(jnp.int32)
            actions = jnp.where(explore | unvisited, random_actions, greedy_actions)

            env_keys = jax.random.split(env_key, env_count)
            (
                next_states,
                next_obs,
                reward,
                terminated,
                truncated,
                valid,
            ) = jax.vmap(env.step)(states, actions, env_keys)
            done = terminated | truncated | (~valid)

            next_state_idx = task.jax_obs_batch_index(next_obs, counter_cap)
            td_target = jnp.where(
                done,
                reward,
                reward + discount_factor * jnp.max(q_table[next_state_idx], axis=-1),
            )
            q_idx = state_idx + (actions,)
            old_values = q_table[q_idx]
            delta = learning_rate * (td_target - old_values)
            q_table = q_table.at[q_idx].add(delta)

            reset_keys = jax.random.split(reset_key, env_count)
            reset_states, reset_obs = jax.vmap(env.reset)(reset_keys)
            states = jax.tree_util.tree_map(
                lambda stepped, reset: _select_by_mask(done, reset, stepped),
                next_states,
                reset_states,
            )
            obs = _select_by_mask(done, reset_obs, next_obs)
            return (q_table, states, obs), None

        (q_table, states, obs), _ = jax.lax.scan(
            step_body,
            (q_table, states, obs),
            step_keys,
        )
        return q_table, states, obs

    return jax.jit(train_chunk)


def _make_evaluate_fn(
    *,
    task: TabularBenchmarkTask,
    eval_episodes: int,
    eval_max_steps: int,
    max_steps: int,
    counter_cap: int,
):
    del max_steps
    env = task.make_jax_env(eval_max_steps)

    def evaluate(q_table, key):
        reset_key, scan_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, eval_episodes)
        states, obs = jax.vmap(env.reset)(reset_keys)
        done = jnp.zeros((eval_episodes,), dtype=jnp.bool_)
        returns = jnp.zeros((eval_episodes,), dtype=jnp.float32)
        step_keys = jax.random.split(scan_key, eval_max_steps)

        def step_body(carry, step_key):
            states, obs, done, returns = carry
            action_key, env_key = jax.random.split(step_key)
            del action_key

            state_idx = task.jax_obs_batch_index(obs, counter_cap)
            actions = jnp.argmax(q_table[state_idx], axis=-1).astype(jnp.int32)
            env_keys = jax.random.split(env_key, eval_episodes)
            (
                next_states,
                next_obs,
                reward,
                terminated,
                truncated,
                valid,
            ) = jax.vmap(env.step)(states, actions, env_keys)
            active = ~done
            step_done = terminated | truncated | (~valid)
            reward = jnp.where(active, reward, jnp.asarray(0.0, dtype=jnp.float32))
            done_next = done | step_done
            states = jax.tree_util.tree_map(
                lambda old, new: _select_by_mask(active, new, old),
                states,
                next_states,
            )
            obs = _select_by_mask(active, next_obs, obs)
            returns = returns + reward
            return (states, obs, done_next, returns), None

        (_, _, _, returns), _ = jax.lax.scan(
            step_body,
            (states, obs, done, returns),
            step_keys,
        )
        return jnp.mean(returns)

    return jax.jit(evaluate)


def _select_by_mask(mask, on_true, on_false):
    while mask.ndim < on_true.ndim:
        mask = mask[..., None]
    return jnp.where(mask, on_true, on_false)


def _block_until_ready(value: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


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


def _infer_target_return(env_results: dict[str, Any]) -> float:
    final_returns = [
        env_result["final_mean_return"] for env_result in env_results.values()
    ]
    best = max(final_returns)
    worst = min(final_returns)
    return float(worst + 0.8 * (best - worst))


def _time_to_target(
    wall_time_s: Sequence[float], returns: Sequence[float], target: float
) -> float | None:
    for time_s, return_value in zip(wall_time_s, returns, strict=True):
        if return_value >= target:
            return float(time_s)
    return None


def _plot_learning_curves(
    result: dict[str, Any], output_path: Path, *, x_key: str
) -> None:
    import matplotlib.pyplot as plt

    task_items = list(result["tasks"].items())
    fig, axes = plt.subplots(
        len(task_items),
        1,
        figsize=(10.8, max(4.0, 3.5 * len(task_items))),
        squeeze=False,
        sharex=False,
    )
    color_map = plt.get_cmap("viridis")

    for ax, (_, task_result) in zip(axes[:, 0], task_items, strict=True):
        env_items = list(task_result["env_counts"].items())
        for idx, (env_count, env_result) in enumerate(env_items):
            color = color_map(idx / max(1, len(env_items) - 1))
            x_values = env_result[f"mean_{x_key}"]
            x_rows = env_result[x_key]
            if x_key == "wall_time_s":
                x_values = _scale_seconds_to_milliseconds(x_values)
                x_rows = [
                    _scale_seconds_to_milliseconds(row) for row in x_rows
                ]
            ax.plot(
                x_values,
                env_result["mean_eval_return"],
                marker="o",
                linewidth=2.1,
                label=f"{env_count} envs",
                color=color,
            )
            _plot_repeat_learning_curves(
                ax=ax,
                x_rows=x_rows,
                y_rows=env_result["eval_return"],
                color=color,
            )
        ax.set_title(task_result["title"])
        ax.set_ylabel("Mean greedy eval return")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best", ncols=2)

    xlabel = (
        "Steady training time (milliseconds; compile/eval excluded)"
        if x_key == "wall_time_s"
        else "Environment transitions"
    )
    axes[-1, 0].set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_repeat_learning_curves(*, ax, x_rows, y_rows, color) -> None:
    if len(x_rows) <= 1:
        return
    for x_values, y_values in zip(x_rows, y_rows, strict=True):
        ax.plot(x_values, y_values, color=color, alpha=0.14, linewidth=1.0)


def _plot_time_to_target(result: dict[str, Any], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    task_items = list(result["tasks"].items())
    fig, axes = plt.subplots(
        len(task_items),
        1,
        figsize=(10.8, max(4.0, 3.2 * len(task_items))),
        squeeze=False,
        sharex=True,
    )

    for ax, (_, task_result) in zip(axes[:, 0], task_items, strict=True):
        env_counts = []
        times = []
        missing_envs = []
        target = task_result["config"]["target_return"]
        for env_count, env_result in task_result["env_counts"].items():
            time_to_target = env_result["time_to_target_s"]
            if time_to_target is None:
                missing_envs.append(int(env_count))
                continue
            env_counts.append(int(env_count))
            times.append(time_to_target * 1000.0)

        if env_counts:
            ax.plot(
                env_counts,
                times,
                marker="o",
                linewidth=2.1,
                color="#4f7f45",
                label="Target reached",
            )
        for env_count in missing_envs:
            ax.scatter(
                env_count,
                0,
                marker="x",
                color="#9a3c3c",
                s=70,
                label="Not reached" if env_count == missing_envs[0] else None,
            )
        ax.set_title(f"{task_result['title']} target return {target:.2f}")
        ax.set_ylabel("Milliseconds")
        ax.set_xscale("log", base=2)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")

    env_counts = result["config"]["env_counts"]
    axes[-1, 0].set_xticks(env_counts)
    axes[-1, 0].set_xticklabels([str(value) for value in env_counts])
    axes[-1, 0].set_xlabel("Parallel environments")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _scale_seconds_to_milliseconds(values: Sequence[float]) -> list[float]:
    return [float(value) * 1000.0 for value in values]


def _without_history(result: dict[str, Any]) -> dict[str, Any]:
    stripped = {key: value for key, value in result.items() if key != "tasks"}
    stripped["tasks"] = {}
    for task_name, task_result in result["tasks"].items():
        stripped["tasks"][task_name] = {
            "title": task_result["title"],
            "config": task_result["config"],
            "env_counts": {
                env_count: {
                    "compile_first_call_s": env_result["compile_first_call_s"],
                    "mean_wall_time_s": env_result["mean_wall_time_s"],
                    "mean_environment_transitions": env_result[
                        "mean_environment_transitions"
                    ],
                    "mean_eval_return": env_result["mean_eval_return"],
                    "final_mean_return": env_result["final_mean_return"],
                    "total_mean_train_time_s": env_result[
                        "total_mean_train_time_s"
                    ],
                    "transitions_per_second": env_result["transitions_per_second"],
                    "time_to_target_s": env_result["time_to_target_s"],
                }
                for env_count, env_result in task_result["env_counts"].items()
            },
        }
    return stripped


def main() -> None:
    """Run the command-line parallel Q-learning scaling benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark learning speed versus JAX parallel env count."
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
    parser.add_argument("--chunks", type=int, default=20)
    parser.add_argument("--updates-per-chunk", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--counter-cap", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--eval-max-steps", type=int, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--plot-output-dir", type=Path, default=None)
    parser.add_argument("--print-history", action="store_true")
    args = parser.parse_args()

    result = run_parallel_learning_benchmark(
        task_names=args.tasks,
        env_counts=args.env_counts,
        chunks=args.chunks,
        updates_per_chunk=args.updates_per_chunk,
        max_steps=args.max_steps,
        repeats=args.repeats,
        seed=args.seed,
        counter_cap=args.counter_cap,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        epsilon=args.epsilon,
        eval_episodes=args.eval_episodes,
        eval_max_steps=args.eval_max_steps,
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
