import argparse
import json
import time
from collections.abc import Callable, Iterable
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise

from examples.jax.function_approximation_speed import (
    BACKENDS,
    CONTINUOUS_ALGORITHMS,
    DISCRETE_ALGORITHMS,
    TASKS,
    algorithm_kwargs,
    load_algorithm,
    make_task_env,
)
from pycrm.automaton import CountingRewardMachine, RmToCrmAdapter

REWARD_CAPS = {
    "rm-discrete": 1020.0,
    "crm-discrete": 40.0,
    "rm-continuous": 1020.0,
    "crm-continuous": 220.0,
}
TARGET_SEQUENCES = {
    "rm-discrete": ("T_1", "T_2", "T_3"),
    "crm-discrete": ("T_1", "T_1", "T_2", "T_3"),
    "rm-continuous": ("T_1", "T_2", "T_3"),
    "crm-continuous": ("T_1",) * 11 + ("T_2",) * 11,
}


class TrainingProgressCallback(BaseCallback):
    """Print progress while an SB3/SBX learn call is running."""

    def __init__(
        self,
        *,
        run_id: str,
        phase: str,
        chunk_start_timesteps: int,
        progress_freq: int,
    ) -> None:
        """Create the progress callback."""
        super().__init__(verbose=0)
        self.run_id = run_id
        self.phase = phase
        self.chunk_start_timesteps = chunk_start_timesteps
        self.progress_freq = progress_freq
        self._started_at = 0.0
        self._last_report_at = 0

    def _on_training_start(self) -> None:
        self._started_at = time.perf_counter()
        self._last_report_at = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_report_at < self.progress_freq:
            return True
        self._last_report_at = self.num_timesteps
        total_timesteps = self.chunk_start_timesteps + self.num_timesteps
        elapsed = time.perf_counter() - self._started_at
        print(
            f"[{self.phase}] {self.run_id}: "
            f"{total_timesteps} train steps (+{self.num_timesteps} this chunk) "
            f"in {elapsed:.1f}s",
            flush=True,
        )
        return True


def run_full_benchmark(
    *,
    tasks: Iterable[str],
    algorithms: Iterable[str],
    backends: Iterable[str],
    seed: int,
    repeats: int,
    target_score: float,
    max_budget_multiplier: float,
    eval_episodes: int,
    baseline_episodes: int,
    required_consecutive: int,
    buffer_size: int,
    batch_size: int,
    learning_starts: int,
    device: str,
    json_output: Path | None = None,
    plot_output_dir: Path | None = None,
    max_steps_override: int | None = None,
    min_budget_override: int | None = None,
    max_budget_override: int | None = None,
    calibration_budget_override: int | None = None,
    eval_freq_override: int | None = None,
    calibration_budget_multiplier: float = 0.5,
    max_calibration_candidates: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Run full-scale function-approximation experiments."""
    selected_tasks = _expand_tasks(tasks)
    selected_backends = _expand_backends(backends)
    result = _load_or_initialize_result(
        json_output=json_output,
        resume=resume,
        config={
            "tasks": selected_tasks,
            "algorithms": list(algorithms),
            "backends": selected_backends,
            "seed": seed,
            "repeats": repeats,
            "target_score": target_score,
            "max_budget_multiplier": max_budget_multiplier,
            "eval_episodes": eval_episodes,
            "baseline_episodes": baseline_episodes,
            "required_consecutive": required_consecutive,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "learning_starts": learning_starts,
            "device": device,
            "max_steps_override": max_steps_override,
            "min_budget_override": min_budget_override,
            "max_budget_override": max_budget_override,
            "calibration_budget_override": calibration_budget_override,
            "eval_freq_override": eval_freq_override,
            "calibration_budget_multiplier": calibration_budget_multiplier,
            "max_calibration_candidates": max_calibration_candidates,
        },
    )

    for task_index, task_name in enumerate(selected_tasks):
        task = TASKS[task_name]
        task_algorithms = _expand_algorithms(algorithms, task["control"])
        max_steps = int(
            task["max_steps"] if max_steps_override is None else max_steps_override
        )
        target = result["targets"].get(task_name)
        if target is None:
            target = compute_task_target(
                task_name=task_name,
                max_steps=max_steps,
                target_score=target_score,
                baseline_episodes=baseline_episodes,
                seed=seed + 1_000_000 + task_index,
            )
            result["targets"][task_name] = target
            _checkpoint(result, json_output, plot_output_dir)

        for algorithm in task_algorithms:
            for backend in selected_backends:
                for repeat in range(repeats):
                    run_id = _run_id(task_name, algorithm, backend, repeat)
                    if _has_final_run(result, run_id):
                        continue

                    run_seed = seed + 100_000 * task_index + 10_000 * repeat
                    budget = _budget_for_task(
                        task_name=task_name,
                        max_budget_multiplier=max_budget_multiplier,
                        calibration_budget_multiplier=calibration_budget_multiplier,
                        min_budget_override=min_budget_override,
                        max_budget_override=max_budget_override,
                        calibration_budget_override=calibration_budget_override,
                        eval_freq_override=eval_freq_override,
                    )
                    calibration = result["calibration"].get(run_id)
                    if calibration is None:
                        print(
                            f"Starting calibration for {run_id}",
                            flush=True,
                        )
                        calibration = run_calibration(
                            task_name=task_name,
                            algorithm=algorithm,
                            backend=backend,
                            target=target,
                            max_steps=max_steps,
                            budget=budget,
                            buffer_size=buffer_size,
                            batch_size=batch_size,
                            learning_starts=learning_starts,
                            device=device,
                            seed=run_seed + 2_000_000,
                            max_candidates=max_calibration_candidates,
                            eval_episodes=eval_episodes,
                        )
                        result["calibration"][run_id] = calibration
                        _checkpoint(result, json_output, plot_output_dir)

                    best_config = select_best_candidate(calibration["candidates"])
                    print(
                        f"Starting final run for {run_id} "
                        f"with calibration score "
                        f"{best_config['final_normalized_score']}",
                        flush=True,
                    )
                    final_run = train_until_converged(
                        run_id=run_id,
                        task_name=task_name,
                        algorithm=algorithm,
                        backend=backend,
                        hyperparams=best_config["hyperparams"],
                        target=target,
                        max_steps=max_steps,
                        min_timesteps=budget["min_timesteps"],
                        max_timesteps=budget["max_timesteps"],
                        eval_freq=budget["eval_freq"],
                        eval_episodes=eval_episodes,
                        required_consecutive=required_consecutive,
                        buffer_size=buffer_size,
                        batch_size=batch_size,
                        learning_starts=learning_starts,
                        device=device,
                        seed=run_seed,
                        repeat=repeat,
                        phase="final",
                    )
                    final_run["selected_calibration"] = best_config
                    result["runs"].append(final_run)
                    result["summary"] = summarize_final_runs(result["runs"])
                    _checkpoint(result, json_output, plot_output_dir)

    result["summary"] = summarize_final_runs(result["runs"])
    _checkpoint(result, json_output, plot_output_dir)
    return result


def compute_task_target(
    *,
    task_name: str,
    max_steps: int,
    target_score: float,
    baseline_episodes: int,
    seed: int,
) -> dict[str, Any]:
    """Compute reward cap and empirical oracle/random targets for one task."""
    reward_cap = compute_reward_cap(task_name)
    random_eval = evaluate_hand_policy(
        task_name=task_name,
        max_steps=max_steps,
        episodes=baseline_episodes,
        seed=seed,
        policy=random_policy,
    )
    oracle_eval = evaluate_hand_policy(
        task_name=task_name,
        max_steps=max_steps,
        episodes=baseline_episodes,
        seed=seed + 100_000,
        policy=oracle_policy,
    )
    denom = max(oracle_eval["mean_return"] - random_eval["mean_return"], 1e-9)
    target_return = random_eval["mean_return"] + target_score * denom
    return {
        "reward_cap": reward_cap,
        "expected_reward_cap": REWARD_CAPS[task_name],
        "target_score": target_score,
        "target_return": float(target_return),
        "oracle": oracle_eval,
        "random": random_eval,
        "oracle_gap": float(denom),
        "oracle_gap_valid": bool(
            oracle_eval["mean_return"] > random_eval["mean_return"]
        ),
    }


def compute_reward_cap(task_name: str) -> float:
    """Compute a task reward cap from the positive target-hit transition sequence."""
    task = TASKS[task_name]
    core = __import__(task["core_module"], fromlist=["PuckWorld"])
    machine = getattr(core, task["machine"])()
    crm = (
        machine
        if isinstance(machine, CountingRewardMachine)
        else RmToCrmAdapter(machine)
    )
    props_by_name = {prop.name: prop for prop in crm.env_prop_enum}
    u = crm.u_0
    c = crm.c_0
    total = 0.0
    for prop_name in TARGET_SEQUENCES[task_name]:
        u, c, reward_fn = crm.transition(u, c, {props_by_name[prop_name]})
        total += float(reward_fn(None, None, None))
    return total


def evaluate_hand_policy(
    *,
    task_name: str,
    max_steps: int,
    episodes: int,
    seed: int,
    policy: Callable[[Any, np.ndarray], Any],
) -> dict[str, Any]:
    """Evaluate a non-learning policy in the same cross-product env."""
    returns = []
    successes = []
    for episode in range(episodes):
        episode_seed = seed + episode
        np.random.seed(episode_seed)
        env = make_task_env(task_name, max_steps, episode_seed)
        env.action_space.seed(episode_seed)
        obs, _ = env.reset(seed=episode_seed)
        terminated = False
        truncated = False
        episode_return = 0.0
        while not (terminated or truncated):
            action = policy(env, obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
        env.close()
        returns.append(episode_return)
        successes.append(bool(terminated and not truncated))
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "returns": [float(value) for value in returns],
        "success_rate": float(np.mean(successes)),
    }


def random_policy(env, obs: np.ndarray):
    """Sample a random action."""
    del obs
    return env.action_space.sample()


def oracle_policy(env, obs: np.ndarray):
    """Pursue the currently relevant PuckWorld target with a PD controller."""
    ground_obs = env.to_ground_obs(obs)
    agent_pos = ground_obs[:2]
    agent_vel = ground_obs[2:4]
    target_pos = _current_target_position(env, ground_obs)
    delta = _wrapped_delta(agent_pos, target_pos)
    control = delta - 1.5 * agent_vel

    if getattr(env.action_space, "n", None) is not None:
        if abs(control[0]) >= abs(control[1]):
            return 0 if control[0] >= 0 else 1
        return 2 if control[1] >= 0 else 3

    action = 0.08 * delta - 0.6 * agent_vel
    return np.clip(action, env.action_space.low, env.action_space.high).astype(
        np.float32
    )


def run_calibration(
    *,
    task_name: str,
    algorithm: str,
    backend: str,
    target: dict[str, Any],
    max_steps: int,
    budget: dict[str, int],
    buffer_size: int,
    batch_size: int,
    learning_starts: int,
    device: str,
    seed: int,
    max_candidates: int | None,
    eval_episodes: int,
) -> dict[str, Any]:
    """Run the fixed calibration sweep and return candidate summaries."""
    candidates = calibration_candidates(
        task_name=task_name,
        algorithm=algorithm,
        backend=backend,
        buffer_size=buffer_size,
        batch_size=batch_size,
        learning_starts=learning_starts,
    )
    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    candidate_results = []
    for index, hyperparams in enumerate(candidates):
        candidate = train_until_converged(
            run_id=f"calibration:{task_name}:{algorithm}:{backend}:{index}",
            task_name=task_name,
            algorithm=algorithm,
            backend=backend,
            hyperparams=hyperparams,
            target=target,
            max_steps=max_steps,
            min_timesteps=0,
            max_timesteps=budget["calibration_timesteps"],
            eval_freq=budget["eval_freq"],
            eval_episodes=eval_episodes,
            required_consecutive=1,
            buffer_size=buffer_size,
            batch_size=batch_size,
            learning_starts=learning_starts,
            device=device,
            seed=seed + index,
            repeat=0,
            phase="calibration",
        )
        candidate_results.append(candidate)
        if candidate["solved"]:
            break
    return {
        "task": task_name,
        "algorithm": algorithm,
        "backend": backend,
        "candidates": candidate_results,
        "best": select_best_candidate(candidate_results),
    }


def calibration_candidates(
    *,
    task_name: str,
    algorithm: str,
    backend: str,
    buffer_size: int,
    batch_size: int,
    learning_starts: int,
) -> list[dict[str, Any]]:
    """Return the deterministic calibration grid for an algorithm."""
    base = base_hyperparams(
        task_name=task_name,
        algorithm=algorithm,
        backend=backend,
        buffer_size=buffer_size,
        batch_size=batch_size,
        learning_starts=learning_starts,
    )
    candidates = [base]
    if algorithm in {"dqn", "cdqn"}:
        exploration_values = sorted({base["exploration_fraction"], 0.2, 0.7})
        for learning_rate, batch, exploration_fraction in product(
            (1e-4, 3e-4, 1e-3), (base["batch_size"], 256, 1024), exploration_values
        ):
            candidates.append(
                {
                    **base,
                    "learning_rate": learning_rate,
                    "batch_size": batch,
                    "exploration_fraction": exploration_fraction,
                }
            )
    elif algorithm in {"ddpg", "cddpg", "td3", "ctd3"}:
        for learning_rate, batch, sigma in product(
            (3e-4, 1e-3), (256, 1024), (0.05, 0.1)
        ):
            candidates.append(
                {
                    **base,
                    "learning_rate": learning_rate,
                    "batch_size": batch,
                    "action_noise_sigma": sigma,
                }
            )
    elif algorithm in {"sac", "csac"}:
        for learning_rate, batch in product((3e-4, 1e-3), (256, 1024)):
            candidates.append(
                {
                    **base,
                    "learning_rate": learning_rate,
                    "batch_size": batch,
                }
            )
    return _dedupe_hyperparams(candidates)


def base_hyperparams(
    *,
    task_name: str,
    algorithm: str,
    backend: str,
    buffer_size: int,
    batch_size: int,
    learning_starts: int,
) -> dict[str, Any]:
    """Return current-example-style hyperparameters for one algorithm."""
    if algorithm in {"dqn", "cdqn"}:
        learning_rate = 1e-4
        exploration_fraction = 0.5 if task_name.startswith("crm-") else 0.1
        exploration_final_eps = 0.1
        target_update_interval = None
        if algorithm == "cdqn" and task_name.startswith("rm-"):
            exploration_fraction = 0.2
        if backend == "sbx" and algorithm in {"dqn", "cdqn"}:
            learning_rate = 3e-4
            batch_size = min(batch_size, 256)
            exploration_fraction = max(exploration_fraction, 0.2)
            exploration_final_eps = 0.05
            target_update_interval = 1000
    elif algorithm in {"sac", "csac"}:
        learning_rate = 3e-4
        exploration_fraction = None
        exploration_final_eps = None
        target_update_interval = None
    else:
        learning_rate = 1e-3
        exploration_fraction = None
        exploration_final_eps = None
        target_update_interval = None

    hyperparams: dict[str, Any] = {
        "learning_rate": learning_rate,
        "buffer_size": buffer_size,
        "batch_size": batch_size,
        "learning_starts": learning_starts,
        "action_noise_sigma": None,
    }
    if exploration_fraction is not None:
        hyperparams["exploration_fraction"] = exploration_fraction
        hyperparams["exploration_final_eps"] = exploration_final_eps
    if target_update_interval is not None:
        hyperparams["target_update_interval"] = target_update_interval
    return hyperparams


def train_until_converged(
    *,
    run_id: str,
    task_name: str,
    algorithm: str,
    backend: str,
    hyperparams: dict[str, Any],
    target: dict[str, Any],
    max_steps: int,
    min_timesteps: int,
    max_timesteps: int,
    eval_freq: int,
    eval_episodes: int,
    required_consecutive: int,
    buffer_size: int,
    batch_size: int,
    learning_starts: int,
    device: str,
    seed: int,
    repeat: int,
    phase: str,
) -> dict[str, Any]:
    """Train one agent in chunks until target or budget exhaustion."""
    np.random.seed(seed)

    def make_env(env_seed: int):
        return make_task_env(task_name, max_steps, env_seed)

    env = make_env(seed)
    model = _make_model(
        task_name=task_name,
        algorithm=algorithm,
        backend=backend,
        env=env,
        hyperparams=hyperparams,
        buffer_size=buffer_size,
        batch_size=batch_size,
        learning_starts=learning_starts,
        device=device,
        seed=seed,
    )

    trained_timesteps = 0
    cumulative_train_time = 0.0
    cumulative_eval_time = 0.0
    curve = []
    solved = False
    time_to_target_s = None
    timesteps_to_target = None

    while trained_timesteps < max_timesteps:
        chunk = min(eval_freq, max_timesteps - trained_timesteps)
        started_at = time.perf_counter()
        model.learn(
            total_timesteps=chunk,
            reset_num_timesteps=trained_timesteps == 0,
            callback=TrainingProgressCallback(
                run_id=run_id,
                phase=phase,
                chunk_start_timesteps=trained_timesteps,
                progress_freq=max(1000, eval_freq // 10),
            ),
            progress_bar=False,
            log_interval=10_000,
        )
        cumulative_train_time += time.perf_counter() - started_at
        trained_timesteps += chunk

        eval_started_at = time.perf_counter()
        eval_result = evaluate_model_policy(
            model=model,
            task_name=task_name,
            max_steps=max_steps,
            episodes=eval_episodes,
            seed=seed + 10_000_000 + trained_timesteps,
        )
        cumulative_eval_time += time.perf_counter() - eval_started_at
        normalized_score = normalize_return(eval_result["mean_return"], target)
        point = {
            "timesteps": int(trained_timesteps),
            "time_s": float(cumulative_train_time),
            "eval_time_s": float(cumulative_eval_time),
            "mean_return": eval_result["mean_return"],
            "std_return": eval_result["std_return"],
            "success_rate": eval_result["success_rate"],
            "normalized_score": float(normalized_score),
            "target_return": target["target_return"],
            "target_score": target["target_score"],
        }
        curve.append(point)
        print(
            f"[{phase}] {run_id}: eval at {trained_timesteps} steps, "
            f"return={point['mean_return']:.3f}, "
            f"score={point['normalized_score']:.3f}, "
            f"target={target['target_score']:.3f}",
            flush=True,
        )

        if normalized_score >= target["target_score"] and time_to_target_s is None:
            time_to_target_s = float(cumulative_train_time)
            timesteps_to_target = int(trained_timesteps)

        if has_converged(
            curve,
            min_timesteps=min_timesteps,
            required_consecutive=required_consecutive,
            target_score=target["target_score"],
        ):
            solved = True
            break

    env.close()
    final_point = curve[-1] if curve else None
    return {
        "run_id": run_id,
        "phase": phase,
        "task": task_name,
        "task_title": TASKS[task_name]["title"],
        "algorithm": algorithm,
        "backend": backend,
        "repeat": repeat,
        "seed": seed,
        "hyperparams": hyperparams,
        "min_timesteps": int(min_timesteps),
        "max_timesteps": int(max_timesteps),
        "trained_timesteps": int(trained_timesteps),
        "eval_freq": int(eval_freq),
        "eval_episodes": int(eval_episodes),
        "train_time_s": float(cumulative_train_time),
        "eval_time_s": float(cumulative_eval_time),
        "steps_per_second": float(trained_timesteps / max(cumulative_train_time, 1e-9)),
        "solved": bool(solved),
        "time_to_target_s": time_to_target_s,
        "timesteps_to_target": timesteps_to_target,
        "final_eval_return": (
            None if final_point is None else final_point["mean_return"]
        ),
        "final_normalized_score": (
            None if final_point is None else final_point["normalized_score"]
        ),
        "returns_curve": curve,
    }


def evaluate_model_policy(
    *,
    model,
    task_name: str,
    max_steps: int,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate a learned model with deterministic actions."""
    returns = []
    successes = []
    for episode in range(episodes):
        episode_seed = seed + episode
        np.random.seed(episode_seed)
        env = make_task_env(task_name, max_steps, episode_seed)
        obs, _ = env.reset(seed=episode_seed)
        terminated = False
        truncated = False
        episode_return = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
        env.close()
        returns.append(episode_return)
        successes.append(bool(terminated and not truncated))
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "returns": [float(value) for value in returns],
        "success_rate": float(np.mean(successes)),
    }


def normalize_return(mean_return: float, target: dict[str, Any]) -> float:
    """Normalize returns against random and oracle empirical baselines."""
    random_mean = float(target["random"]["mean_return"])
    return float((mean_return - random_mean) / max(float(target["oracle_gap"]), 1e-9))


def has_converged(
    curve: list[dict[str, Any]],
    *,
    min_timesteps: int,
    required_consecutive: int,
    target_score: float,
) -> bool:
    """Check the adaptive convergence rule."""
    if len(curve) < required_consecutive:
        return False
    if int(curve[-1]["timesteps"]) < min_timesteps:
        return False
    recent = curve[-required_consecutive:]
    return all(float(point["normalized_score"]) >= target_score for point in recent)


def select_best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the best calibration candidate."""
    if not candidates:
        raise ValueError("Cannot select from an empty candidate list.")
    return max(
        candidates,
        key=lambda candidate: (
            bool(candidate["solved"]),
            float(candidate["final_normalized_score"] or float("-inf")),
            -float(candidate["train_time_s"]),
        ),
    )


def summarize_final_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize final runs for timing and speedup plots."""
    rows = []
    for run in sorted(
        runs, key=lambda item: (item["task"], item["algorithm"], item["backend"])
    ):
        rows.append(
            {
                "run_id": run["run_id"],
                "task": run["task"],
                "algorithm": run["algorithm"],
                "backend": run["backend"],
                "train_time_s": run["train_time_s"],
                "steps_per_second": run["steps_per_second"],
                "final_eval_return": run["final_eval_return"],
                "final_normalized_score": run["final_normalized_score"],
                "solved": run["solved"],
                "time_to_target_s": run["time_to_target_s"],
                "timesteps_to_target": run["timesteps_to_target"],
            }
        )

    sb3_times = {
        (row["task"], row["algorithm"]): row["train_time_s"]
        for row in rows
        if row["backend"] == "sb3"
    }
    for row in rows:
        if row["backend"] == "sbx":
            key = (row["task"], row["algorithm"])
            if key in sb3_times:
                row["speedup_vs_sb3"] = float(sb3_times[key] / row["train_time_s"])
    return rows


def write_plots(result: dict[str, Any], output_dir: Path) -> dict[str, str]:
    """Write the full-scale plot bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "returns_curve": output_dir / "function_approx_returns_time.png",
        "normalized_returns_curve": (
            output_dir / "function_approx_normalized_returns_time.png"
        ),
        "times": output_dir / "function_approx_train_times.png",
        "speedups": output_dir / "function_approx_sbx_speedups.png",
    }
    _plot_returns(result, paths["returns_curve"], normalized=False)
    _plot_returns(result, paths["normalized_returns_curve"], normalized=True)
    _plot_times(result, paths["times"])
    _plot_speedups(result, paths["speedups"])
    return {key: str(value) for key, value in paths.items()}


def _plot_returns(result: dict[str, Any], path: Path, *, normalized: bool) -> None:
    import matplotlib.pyplot as plt

    runs = result["runs"]
    if not runs:
        _write_empty_plot(path, "No final runs yet")
        return

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

    for ax, task_name in zip(axes_array, task_names, strict=False):
        target = result["targets"][task_name]
        for run in [item for item in runs if item["task"] == task_name]:
            curve = run["returns_curve"]
            if not curve:
                continue
            y_key = "normalized_score" if normalized else "mean_return"
            y_values = [point[y_key] for point in curve]
            ax.plot(
                [point["time_s"] for point in curve],
                y_values,
                label=f"{run['backend'].upper()} {run['algorithm'].upper()}",
                color=colors[run["backend"]],
                linestyle=linestyles.get(run["algorithm"], "-"),
                alpha=0.85,
            )
            if not run["solved"]:
                ax.scatter(
                    curve[-1]["time_s"],
                    y_values[-1],
                    marker="x",
                    color=colors[run["backend"]],
                )
        target_line = target["target_score"] if normalized else target["target_return"]
        ax.axhline(target_line, color="#222222", linewidth=1.0, alpha=0.75)
        ax.set_title(TASKS[task_name]["title"])
        ax.set_xlabel("Wall-clock training time (s)")
        ax.set_ylabel("Normalized score" if normalized else "Mean evaluation return")
        ax.grid(alpha=0.25)
        ax.legend(fontsize="small", ncols=2)

    for ax in axes_array[len(task_names) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_times(result: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = result["summary"]
    if not rows:
        _write_empty_plot(path, "No final runs yet")
        return
    labels = [
        f"{row['task']}\n{row['algorithm'].upper()} {row['backend'].upper()}"
        for row in rows
    ]
    values = [row["train_time_s"] for row in rows]
    colors = ["#4C78A8" if row["backend"] == "sb3" else "#F58518" for row in rows]

    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(labels)), 5))
    bars = ax.bar(range(len(labels)), values, color=colors)
    for bar, row in zip(bars, rows, strict=True):
        if not row["solved"]:
            bar.set_hatch("//")
    ax.set_ylabel("Train time (s)")
    ax.set_xticks(range(len(labels)), labels, rotation=65, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_speedups(result: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = [row for row in result["summary"] if "speedup_vs_sb3" in row]
    if not rows:
        _write_empty_plot(path, "No SBX/SB3 pairs yet")
        return
    labels = [f"{row['task']}\n{row['algorithm'].upper()}" for row in rows]
    values = [row["speedup_vs_sb3"] for row in rows]

    fig, ax = plt.subplots(figsize=(max(8, 0.65 * len(labels)), 5))
    ax.axhline(1.0, color="#222222", linewidth=1.0)
    bars = ax.bar(range(len(labels)), values, color="#F58518")
    for bar, row in zip(bars, rows, strict=True):
        if not row["solved"]:
            bar.set_hatch("//")
    ax.set_ylabel("SB3 time / SBX time")
    ax.set_xticks(range(len(labels)), labels, rotation=65, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _make_model(
    *,
    task_name: str,
    algorithm: str,
    backend: str,
    env,
    hyperparams: dict[str, Any],
    buffer_size: int,
    batch_size: int,
    learning_starts: int,
    device: str,
    seed: int,
):
    algorithm_class = load_algorithm(backend, algorithm)
    kwargs = algorithm_kwargs(
        task_name=task_name,
        algorithm=algorithm,
        backend=backend,
        env=env,
        buffer_size=int(hyperparams.get("buffer_size", buffer_size)),
        batch_size=int(hyperparams.get("batch_size", batch_size)),
        learning_starts=int(hyperparams.get("learning_starts", learning_starts)),
        device=device,
        seed=seed,
    )
    kwargs["learning_rate"] = float(hyperparams["learning_rate"])
    if "exploration_fraction" in hyperparams:
        kwargs["exploration_fraction"] = float(hyperparams["exploration_fraction"])
        kwargs["exploration_final_eps"] = float(
            hyperparams.get("exploration_final_eps", 0.1)
        )
    if "target_update_interval" in hyperparams:
        kwargs["target_update_interval"] = int(hyperparams["target_update_interval"])
    sigma = hyperparams.get("action_noise_sigma")
    if sigma is not None and algorithm in {"ddpg", "cddpg", "td3", "ctd3"}:
        action_dim = int(env.action_space.shape[0])
        kwargs["action_noise"] = NormalActionNoise(
            mean=np.zeros(action_dim),
            sigma=float(sigma) * np.ones(action_dim),
        )
    return algorithm_class(**kwargs)


def _budget_for_task(
    *,
    task_name: str,
    max_budget_multiplier: float,
    calibration_budget_multiplier: float,
    min_budget_override: int | None,
    max_budget_override: int | None,
    calibration_budget_override: int | None,
    eval_freq_override: int | None,
) -> dict[str, int]:
    original = int(TASKS[task_name]["example_timesteps"])
    min_timesteps = original if min_budget_override is None else min_budget_override
    max_timesteps = (
        int(original * max_budget_multiplier)
        if max_budget_override is None
        else max_budget_override
    )
    calibration_timesteps = (
        int(original * calibration_budget_multiplier)
        if calibration_budget_override is None
        else calibration_budget_override
    )
    eval_freq = (
        max(10_000, original // 25)
        if eval_freq_override is None
        else eval_freq_override
    )
    return {
        "original_timesteps": original,
        "min_timesteps": int(min_timesteps),
        "max_timesteps": int(max_timesteps),
        "calibration_timesteps": int(calibration_timesteps),
        "eval_freq": int(eval_freq),
    }


def _current_target_position(env, ground_obs: np.ndarray) -> np.ndarray:
    target_index = 0
    if env.u == 1:
        target_index = 1
    elif env.u == 2:
        target_index = 2
    start = 4 + 2 * target_index
    return ground_obs[start : start + 2]


def _wrapped_delta(agent_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    delta = target_pos - agent_pos
    return ((delta + 1.0) % 2.0) - 1.0


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


def _dedupe_hyperparams(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for candidate in candidates:
        key = json.dumps(candidate, sort_keys=True)
        if key not in seen:
            deduped.append(candidate)
            seen.add(key)
    return deduped


def _load_or_initialize_result(
    *,
    json_output: Path | None,
    resume: bool,
    config: dict[str, Any],
) -> dict[str, Any]:
    if resume and json_output is not None and json_output.exists():
        return json.loads(json_output.read_text(encoding="utf-8"))
    return {
        "config": config,
        "targets": {},
        "calibration": {},
        "runs": [],
        "summary": [],
        "plots": {},
        "note": (
            "Full-scale function approximation timings use Gymnasium/Python "
            "PuckWorld envs. SBX accelerates neural-network updates with JAX."
        ),
    }


def _checkpoint(
    result: dict[str, Any],
    json_output: Path | None,
    plot_output_dir: Path | None,
) -> None:
    result["summary"] = summarize_final_runs(result["runs"])
    if plot_output_dir is not None:
        result["plots"] = write_plots(result, plot_output_dir)
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(result, indent=2), encoding="utf-8")


def _write_empty_plot(path: Path, message: str) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _has_final_run(result: dict[str, Any], run_id: str) -> bool:
    return any(run["run_id"] == run_id for run in result["runs"])


def _run_id(task: str, algorithm: str, backend: str, repeat: int) -> str:
    return f"{task}|{algorithm}|{backend}|{repeat}"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run full-scale PyCRM function-approximation experiments."
    )
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--algorithms", nargs="+", default=["all"])
    parser.add_argument("--backends", nargs="+", default=["sb3", "sbx"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--target-score", type=float, default=0.90)
    parser.add_argument("--max-budget-multiplier", type=float, default=5.0)
    parser.add_argument("--calibration-budget-multiplier", type=float, default=0.5)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--baseline-episodes", type=int, default=20)
    parser.add_argument("--required-consecutive", type=int, default=3)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=2_500)
    parser.add_argument("--learning-starts", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--min-budget-override", type=int, default=None)
    parser.add_argument("--max-budget-override", type=int, default=None)
    parser.add_argument("--calibration-budget-override", type=int, default=None)
    parser.add_argument("--eval-freq-override", type=int, default=None)
    parser.add_argument("--max-calibration-candidates", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("results/jax_function_approximation_full_scale/benchmark.json"),
    )
    parser.add_argument(
        "--plot-output-dir",
        type=Path,
        default=Path("results/jax_function_approximation_full_scale"),
    )
    return parser.parse_args()


def main() -> None:
    """Run the full-scale benchmark CLI."""
    args = parse_args()
    result = run_full_benchmark(
        tasks=args.tasks,
        algorithms=args.algorithms,
        backends=args.backends,
        seed=args.seed,
        repeats=args.repeats,
        target_score=args.target_score,
        max_budget_multiplier=args.max_budget_multiplier,
        calibration_budget_multiplier=args.calibration_budget_multiplier,
        eval_episodes=args.eval_episodes,
        baseline_episodes=args.baseline_episodes,
        required_consecutive=args.required_consecutive,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        device=args.device,
        max_steps_override=args.max_steps,
        min_budget_override=args.min_budget_override,
        max_budget_override=args.max_budget_override,
        calibration_budget_override=args.calibration_budget_override,
        eval_freq_override=args.eval_freq_override,
        max_calibration_candidates=args.max_calibration_candidates,
        resume=not args.no_resume,
        json_output=args.json_output,
        plot_output_dir=args.plot_output_dir,
    )
    print(f"Wrote benchmark JSON to {args.json_output}")
    for key, path in result["plots"].items():
        print(f"Wrote {key} plot to {path}")
    for row in result["summary"]:
        status = "solved" if row["solved"] else "unsolved"
        speedup = row.get("speedup_vs_sb3")
        suffix = "" if speedup is None else f", speedup vs SB3={speedup:.2f}x"
        print(
            f"{row['task']} {row['algorithm']} {row['backend']}: "
            f"{row['train_time_s']:.3f}s, {status}{suffix}"
        )


if __name__ == "__main__":
    main()
