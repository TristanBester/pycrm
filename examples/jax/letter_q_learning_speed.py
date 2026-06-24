import argparse
import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from examples.introduction.core.crossproduct import LetterWorldCrossProduct
from examples.introduction.core.ground import LetterWorld
from examples.introduction.core.label import LetterWorldLabellingFunction
from examples.introduction.core.machine import LetterWorldCountingRewardMachine
from pycrm.jax import JaxCrossProduct, compile_crm

N_ACTIONS = 4
SYMBOL_STATES = 2
N_ROWS = LetterWorld.N_ROWS
N_COLS = LetterWorld.N_COLS


def run_benchmark(
    *,
    episodes: int = 5000,
    max_steps: int = 500,
    repeats: int = 3,
    seed: int = 0,
    counter_cap: int = 12,
    learning_rate: float = 0.1,
    discount_factor: float = 0.99,
    epsilon: float = 0.1,
) -> dict[str, Any]:
    """Run Python and JAX Letter World Q-learning benchmarks."""
    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    if repeats <= 0:
        raise ValueError("repeats must be positive.")
    if counter_cap < 0:
        raise ValueError("counter_cap must be non-negative.")

    training_config = {
        "episodes": episodes,
        "max_steps": max_steps,
        "counter_cap": counter_cap,
        "learning_rate": learning_rate,
        "discount_factor": discount_factor,
        "epsilon": epsilon,
    }
    python_times, python_returns = _time_python_training(
        _run_python_q_learning,
        repeats=repeats,
        seed=seed,
        **training_config,
    )
    python_cql_times, python_cql_returns = _time_python_training(
        _run_python_counterfactual_q_learning,
        repeats=repeats,
        seed=seed + 1_000,
        **training_config,
    )

    train_jit = _make_jax_training_fn(
        counterfactual=False,
        episodes=episodes,
        max_steps=max_steps,
        counter_cap=counter_cap,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
    )
    compile_first_call_s, jax_times, jax_returns = _time_jax_training(
        train_jit, repeats=repeats, seed=seed
    )

    train_cql_jit = _make_jax_training_fn(
        counterfactual=True,
        episodes=episodes,
        max_steps=max_steps,
        counter_cap=counter_cap,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
    )
    cql_compile_first_call_s, jax_cql_times, jax_cql_returns = _time_jax_training(
        train_cql_jit, repeats=repeats, seed=seed + 1_000
    )

    python_mean_s = float(np.mean(python_times))
    python_cql_mean_s = float(np.mean(python_cql_times))
    jax_mean_s = float(np.mean(jax_times))
    jax_cql_mean_s = float(np.mean(jax_cql_times))
    return {
        "config": {
            "episodes": episodes,
            "max_steps": max_steps,
            "repeats": repeats,
            "seed": seed,
            "counter_cap": counter_cap,
            "learning_rate": learning_rate,
            "discount_factor": discount_factor,
            "epsilon": epsilon,
        },
        "python": {
            "label": "Python Q-learning",
            "times_s": [float(t) for t in python_times],
            "mean_s": python_mean_s,
            **_return_summary(python_returns[-1]),
        },
        "python_counterfactual": {
            "label": "Python counterfactual Q-learning",
            "times_s": [float(t) for t in python_cql_times],
            "mean_s": python_cql_mean_s,
            **_return_summary(python_cql_returns[-1]),
        },
        "jax": {
            "label": "JAX Q-learning",
            "compile_first_call_s": float(compile_first_call_s),
            "steady_times_s": [float(t) for t in jax_times],
            "steady_mean_s": jax_mean_s,
            **_return_summary(jax_returns[-1]),
        },
        "jax_counterfactual": {
            "label": "JAX counterfactual Q-learning",
            "compile_first_call_s": float(cql_compile_first_call_s),
            "steady_times_s": [float(t) for t in jax_cql_times],
            "steady_mean_s": jax_cql_mean_s,
            **_return_summary(jax_cql_returns[-1]),
        },
        "history": {
            "python_returns": [
                [float(value) for value in returns] for returns in python_returns
            ],
            "python_counterfactual_returns": [
                [float(value) for value in returns] for returns in python_cql_returns
            ],
            "jax_returns": [
                [float(value) for value in returns] for returns in jax_returns
            ],
            "jax_counterfactual_returns": [
                [float(value) for value in returns] for returns in jax_cql_returns
            ],
        },
        "speedup": {
            "q_learning_steady_state_ratio": (
                python_mean_s / jax_mean_s if jax_mean_s > 0 else float("inf")
            ),
            "counterfactual_steady_state_ratio": (
                python_cql_mean_s / jax_cql_mean_s
                if jax_cql_mean_s > 0
                else float("inf")
            ),
            "steady_state_ratio": (
                python_mean_s / jax_mean_s if jax_mean_s > 0 else float("inf")
            ),
            "note": "JAX compile time is reported separately from steady-state time.",
        },
    }


def save_plots(
    result: dict[str, Any], output_dir: Path, smoothing_window: int = 50
) -> dict[str, str]:
    """Save returns and speedup plots for a benchmark result."""
    if smoothing_window <= 0:
        raise ValueError("smoothing_window must be positive.")

    output_dir.mkdir(parents=True, exist_ok=True)
    returns_path = output_dir / "letter_q_learning_returns.png"
    speedup_path = output_dir / "letter_q_learning_speedup.png"

    _plot_returns(result, returns_path, smoothing_window)
    _plot_speedup(result, speedup_path)

    return {
        "returns_curve": str(returns_path),
        "speedup": str(speedup_path),
    }


def _time_python_training(
    training_fn,
    *,
    repeats: int,
    seed: int,
    episodes: int,
    max_steps: int,
    counter_cap: int,
    learning_rate: float,
    discount_factor: float,
    epsilon: float,
) -> tuple[list[float], list[np.ndarray]]:
    times = []
    returns_history = []

    for repeat_idx in range(repeats):
        start = time.perf_counter()
        returns = training_fn(
            episodes=episodes,
            max_steps=max_steps,
            seed=seed + repeat_idx,
            counter_cap=counter_cap,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
        )
        times.append(time.perf_counter() - start)
        returns_history.append(returns)

    return times, returns_history


def _time_jax_training(train_jit, *, repeats: int, seed: int):
    compile_key = jax.random.PRNGKey(seed)
    start = time.perf_counter()
    _, compile_returns = train_jit(compile_key)
    compile_returns.block_until_ready()
    compile_first_call_s = time.perf_counter() - start

    times = []
    returns_history = []
    for repeat_idx in range(repeats):
        key = jax.random.PRNGKey(seed + 10_000 + repeat_idx)
        start = time.perf_counter()
        _, returns = train_jit(key)
        returns.block_until_ready()
        times.append(time.perf_counter() - start)
        returns_history.append(np.asarray(returns))

    return float(compile_first_call_s), times, returns_history


def _run_python_q_learning(
    *,
    episodes: int,
    max_steps: int,
    seed: int,
    counter_cap: int,
    learning_rate: float,
    discount_factor: float,
    epsilon: float,
) -> np.ndarray:
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    ground_env = LetterWorld()
    crm = LetterWorldCountingRewardMachine()
    cross_product = LetterWorldCrossProduct(
        ground_env=ground_env,
        crm=crm,
        lf=LetterWorldLabellingFunction(),
        max_steps=max_steps,
    )
    q_table = np.zeros(_q_shape(crm, counter_cap), dtype=np.float32)
    returns = np.zeros(episodes, dtype=np.float32)

    for episode in range(episodes):
        obs, _ = cross_product.reset()
        done = False
        episode_return = 0.0

        while not done:
            state_idx = _python_obs_index(obs, counter_cap)
            q_values = q_table[state_idx]
            if rng.random() < epsilon or np.all(q_values == 0):
                action = int(rng.integers(N_ACTIONS))
            else:
                action = int(np.argmax(q_values))

            next_obs, reward, terminated, truncated, _ = cross_product.step(action)
            done = terminated or truncated
            episode_return += reward

            next_state_idx = _python_obs_index(next_obs, counter_cap)
            td_target = reward
            if not done:
                td_target += discount_factor * float(np.max(q_table[next_state_idx]))

            old_value = q_table[state_idx + (action,)]
            q_table[state_idx + (action,)] = old_value + learning_rate * (
                td_target - old_value
            )
            obs = next_obs

        returns[episode] = episode_return

    return returns


def _run_python_counterfactual_q_learning(
    *,
    episodes: int,
    max_steps: int,
    seed: int,
    counter_cap: int,
    learning_rate: float,
    discount_factor: float,
    epsilon: float,
) -> np.ndarray:
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    ground_env = LetterWorld()
    crm = LetterWorldCountingRewardMachine()
    cross_product = LetterWorldCrossProduct(
        ground_env=ground_env,
        crm=crm,
        lf=LetterWorldLabellingFunction(),
        max_steps=max_steps,
    )
    q_table = np.zeros(_q_shape(crm, counter_cap), dtype=np.float32)
    returns = np.zeros(episodes, dtype=np.float32)

    for episode in range(episodes):
        obs, _ = cross_product.reset()
        done = False
        episode_return = 0.0

        while not done:
            state_idx = _python_obs_index(obs, counter_cap)
            q_values = q_table[state_idx]
            if rng.random() < epsilon or np.all(q_values == 0):
                action = int(rng.integers(N_ACTIONS))
            else:
                action = int(np.argmax(q_values))

            next_obs, reward, terminated, truncated, _ = cross_product.step(action)
            done = terminated or truncated
            episode_return += reward

            for cf_obs, cf_action, cf_next_obs, cf_reward, cf_done, _ in zip(
                *cross_product.generate_counterfactual_experience(
                    cross_product.to_ground_obs(obs),
                    action,
                    cross_product.to_ground_obs(next_obs),
                ),
                strict=True,
            ):
                cf_state_idx = _python_obs_index(cf_obs, counter_cap)
                cf_next_state_idx = _python_obs_index(cf_next_obs, counter_cap)
                cf_action = int(cf_action)
                cf_td_target = float(cf_reward)
                if not cf_done:
                    cf_td_target += discount_factor * float(
                        np.max(q_table[cf_next_state_idx])
                    )

                old_cf_value = q_table[cf_state_idx + (cf_action,)]
                q_table[cf_state_idx + (cf_action,)] = old_cf_value + learning_rate * (
                    cf_td_target - old_cf_value
                )

            next_state_idx = _python_obs_index(next_obs, counter_cap)
            td_target = reward
            if not done:
                td_target += discount_factor * float(np.max(q_table[next_state_idx]))

            old_value = q_table[state_idx + (action,)]
            q_table[state_idx + (action,)] = old_value + learning_rate * (
                td_target - old_value
            )
            obs = next_obs

        returns[episode] = episode_return

    return returns


def _make_jax_training_fn(
    *,
    counterfactual: bool,
    episodes: int,
    max_steps: int,
    counter_cap: int,
    learning_rate: float,
    discount_factor: float,
    epsilon: float,
):
    crm = LetterWorldCountingRewardMachine()
    env = JaxCrossProduct(
        compiled_crm=compile_crm(crm),
        reset_fn=_jax_letter_reset,
        step_fn=_jax_letter_step,
        label_fn=_jax_letter_label,
        max_steps=max_steps,
        obs_fn=_jax_letter_obs,
    )
    q_shape = _q_shape(crm, counter_cap)

    def train(key):
        q_table = jnp.zeros(q_shape, dtype=jnp.float32)
        episode_keys = jax.random.split(key, episodes)

        def episode_body(q_carry, episode_key):
            reset_key, step_key = jax.random.split(episode_key)
            state, obs = env.reset(reset_key)
            step_keys = jax.random.split(step_key, max_steps)
            initial_carry = (
                q_carry,
                state,
                obs,
                jnp.asarray(False),
                jnp.asarray(0.0, dtype=jnp.float32),
            )

            def step_body(carry, key_for_step):
                def active_step(active_carry):
                    q_table, state, obs, done, episode_return = active_carry
                    del done

                    eps_key, action_key, env_key = jax.random.split(key_for_step, 3)
                    state_idx = _jax_obs_index(obs, counter_cap)
                    q_values = q_table[state_idx]
                    explore = jax.random.uniform(eps_key) < epsilon
                    unvisited = jnp.all(q_values == 0)
                    random_action = jax.random.randint(
                        action_key, (), 0, N_ACTIONS, dtype=jnp.int32
                    )
                    greedy_action = jnp.argmax(q_values).astype(jnp.int32)
                    action = jnp.where(
                        explore | unvisited, random_action, greedy_action
                    )

                    (
                        next_state,
                        next_obs,
                        reward,
                        terminated,
                        truncated,
                        valid,
                    ) = env.step(state, action, env_key)

                    next_done = terminated | truncated | (~valid)
                    if counterfactual:
                        q_table = _jax_counterfactual_update(
                            env=env,
                            q_table=q_table,
                            ground_obs=state.ground_obs,
                            action=action,
                            next_ground_obs=next_state.ground_obs,
                            counter_cap=counter_cap,
                            learning_rate=learning_rate,
                            discount_factor=discount_factor,
                        )

                    next_state_idx = _jax_obs_index(next_obs, counter_cap)
                    td_target = jnp.where(
                        next_done,
                        reward,
                        reward + discount_factor * jnp.max(q_table[next_state_idx]),
                    )
                    q_idx = state_idx + (action,)
                    old_value = q_table[q_idx]
                    q_table = q_table.at[q_idx].set(
                        old_value + learning_rate * (td_target - old_value)
                    )
                    return (
                        q_table,
                        next_state,
                        next_obs,
                        next_done,
                        episode_return + reward,
                    )

                carry = jax.lax.cond(carry[3], lambda x: x, active_step, carry)
                return carry, None

            final_carry, _ = jax.lax.scan(step_body, initial_carry, step_keys)
            q_next, _, _, _, episode_return = final_carry
            return q_next, episode_return

        return jax.lax.scan(episode_body, q_table, episode_keys)

    return jax.jit(train)


def _jax_counterfactual_update(
    *,
    env: JaxCrossProduct,
    q_table,
    ground_obs,
    action,
    next_ground_obs,
    counter_cap: int,
    learning_rate: float,
    discount_factor: float,
):
    (
        cf_obs,
        cf_actions,
        cf_next_obs,
        cf_rewards,
        cf_done,
        cf_valid,
    ) = env.generate_counterfactual_experience(ground_obs, action, next_ground_obs)

    cf_state_idx = _jax_obs_batch_index(cf_obs, counter_cap)
    cf_next_state_idx = _jax_obs_batch_index(cf_next_obs, counter_cap)
    q_idx = cf_state_idx + (cf_actions,)
    old_values = q_table[q_idx]
    td_target = jnp.where(
        cf_done,
        cf_rewards,
        cf_rewards + discount_factor * jnp.max(q_table[cf_next_state_idx], axis=-1),
    )
    updated_values = old_values + learning_rate * (td_target - old_values)
    updated_values = jnp.where(cf_valid, updated_values, old_values)
    return q_table.at[q_idx].set(updated_values)


def _jax_letter_reset(key):
    del key
    return jnp.asarray([0, 1, 3], dtype=jnp.int32)


def _jax_letter_step(obs, action, key):
    symbol_seen, row, col = obs

    row_next = jnp.select(
        [action == LetterWorld.UP, action == LetterWorld.DOWN],
        [jnp.maximum(row - 1, 0), jnp.minimum(row + 1, N_ROWS - 1)],
        default=row,
    )
    col_next = jnp.select(
        [action == LetterWorld.RIGHT, action == LetterWorld.LEFT],
        [jnp.minimum(col + 1, N_COLS - 1), jnp.maximum(col - 1, 0)],
        default=col,
    )
    at_a = (row_next == 1) & (col_next == 1)
    observes_b = jax.random.bernoulli(key, 0.5).astype(jnp.int32)
    symbol_next = jnp.where((symbol_seen == 0) & at_a, observes_b, symbol_seen)
    return jnp.asarray([symbol_next, row_next, col_next], dtype=jnp.int32)


def _jax_letter_label(obs, action, next_obs):
    del obs, action
    symbol_seen, row, col = next_obs
    at_ab = (row == 1) & (col == 1)
    at_c = (row == 1) & (col == 5)
    return jnp.asarray(
        [
            (symbol_seen == 0) & at_ab,
            (symbol_seen == 1) & at_ab,
            (symbol_seen == 1) & at_c,
        ],
        dtype=jnp.bool_,
    )


def _jax_letter_obs(ground_obs, u, c):
    return jnp.asarray(
        [ground_obs[0], ground_obs[1], ground_obs[2], u, c[0]], dtype=jnp.int32
    )


def _q_shape(
    crm: LetterWorldCountingRewardMachine, counter_cap: int
) -> tuple[int, int, int, int, int, int]:
    num_machine_states = max([crm.u_0, *crm.U, *crm.F]) + 1
    return (
        SYMBOL_STATES,
        N_ROWS,
        N_COLS,
        num_machine_states,
        counter_cap + 1,
        N_ACTIONS,
    )


def _python_obs_index(
    obs: np.ndarray, counter_cap: int
) -> tuple[int, int, int, int, int]:
    symbol_seen, row, col, u, c = [int(value) for value in obs]
    return (
        symbol_seen,
        row,
        col,
        u,
        min(max(c, 0), counter_cap),
    )


def _jax_obs_index(obs, counter_cap: int):
    c = jnp.clip(obs[4], 0, counter_cap)
    return obs[0], obs[1], obs[2], obs[3], c


def _jax_obs_batch_index(obs, counter_cap: int):
    c = jnp.clip(obs[:, 4], 0, counter_cap)
    return obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3], c


def _return_summary(returns: np.ndarray) -> dict[str, float]:
    recent_window = min(50, len(returns))
    return {
        "final_return": float(returns[-1]),
        "recent_average_return": float(np.mean(returns[-recent_window:])),
    }


def _plot_returns(
    result: dict[str, Any], output_path: Path, smoothing_window: int
) -> None:
    import matplotlib.pyplot as plt

    python_returns = np.asarray(result["history"]["python_returns"], dtype=np.float32)
    python_cql_returns = np.asarray(
        result["history"]["python_counterfactual_returns"], dtype=np.float32
    )
    jax_returns = np.asarray(result["history"]["jax_returns"], dtype=np.float32)
    jax_cql_returns = np.asarray(
        result["history"]["jax_counterfactual_returns"], dtype=np.float32
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_return_family(
        ax=ax,
        returns=python_returns,
        label="Python Q-learning",
        color="#2f6fbb",
        smoothing_window=smoothing_window,
    )
    _plot_return_family(
        ax=ax,
        returns=python_cql_returns,
        label="Python CQL",
        color="#4f7f45",
        smoothing_window=smoothing_window,
    )
    _plot_return_family(
        ax=ax,
        returns=jax_returns,
        label="JAX Q-learning",
        color="#c95f2d",
        smoothing_window=smoothing_window,
    )
    _plot_return_family(
        ax=ax,
        returns=jax_cql_returns,
        label="JAX CQL",
        color="#7b4fa3",
        smoothing_window=smoothing_window,
    )
    ax.set_title("Letter World Q-learning and CQL Returns")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode return")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_return_family(
    *,
    ax,
    returns: np.ndarray,
    label: str,
    color: str,
    smoothing_window: int,
) -> None:
    smoothed = np.asarray(
        [_moving_average(row, smoothing_window) for row in returns], dtype=np.float32
    )
    offset = returns.shape[1] - smoothed.shape[1]
    x = np.arange(offset + 1, returns.shape[1] + 1)

    if len(smoothed) > 1:
        for row in smoothed:
            ax.plot(x, row, color=color, alpha=0.18, linewidth=1)

    ax.plot(x, np.mean(smoothed, axis=0), color=color, linewidth=2.4, label=label)


def _plot_speedup(result: dict[str, Any], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    python_times = np.asarray(result["python"]["times_s"], dtype=np.float32)
    jax_times = np.asarray(result["jax"]["steady_times_s"], dtype=np.float32)
    python_cql_times = np.asarray(
        result["python_counterfactual"]["times_s"], dtype=np.float32
    )
    jax_cql_times = np.asarray(
        result["jax_counterfactual"]["steady_times_s"], dtype=np.float32
    )
    per_repeat_speedups = python_times / jax_times
    per_repeat_cql_speedups = python_cql_times / jax_cql_times
    q_steady_speedup = result["speedup"]["q_learning_steady_state_ratio"]
    cql_steady_speedup = result["speedup"]["counterfactual_steady_state_ratio"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    x = np.arange(2)
    width = 0.25
    axes[0].bar(
        x - width,
        [result["python"]["mean_s"], result["python_counterfactual"]["mean_s"]],
        width,
        label="Python",
        color="#2f6fbb",
    )
    axes[0].bar(
        x,
        [result["jax"]["steady_mean_s"], result["jax_counterfactual"]["steady_mean_s"]],
        width,
        label="JAX steady",
        color="#c95f2d",
    )
    axes[0].bar(
        x + width,
        [
            result["jax"]["compile_first_call_s"],
            result["jax_counterfactual"]["compile_first_call_s"],
        ],
        width,
        label="JAX compile+run",
        color="#767676",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Q-learning", "CQL"])
    axes[0].set_title("Benchmark Time")
    axes[0].set_ylabel("Seconds")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)
    axes[0].legend()

    repeat_x = np.arange(1, len(per_repeat_speedups) + 1)
    axes[1].plot(
        repeat_x,
        per_repeat_speedups,
        marker="o",
        color="#4f7f45",
        label="Q-learning repeats",
    )
    axes[1].plot(
        repeat_x,
        per_repeat_cql_speedups,
        marker="o",
        color="#7b4fa3",
        label="CQL repeats",
    )
    axes[1].axhline(
        q_steady_speedup,
        color="#4f7f45",
        linestyle="--",
        linewidth=1.5,
        label=f"Q mean {q_steady_speedup:.2f}x",
    )
    axes[1].axhline(
        cql_steady_speedup,
        color="#7b4fa3",
        linestyle="--",
        linewidth=1.5,
        label=f"CQL mean {cql_steady_speedup:.2f}x",
    )
    axes[1].set_title("JAX Steady-State Speedup")
    axes[1].set_xlabel("Repeat")
    axes[1].set_ylabel("Python / JAX time")
    axes[1].set_xticks(repeat_x)
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode="valid")


def _without_history(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "history"}


def main() -> None:
    """Run the command-line benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark Python and JAX Q-learning in Letter World."
    )
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--counter-cap", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--plot-output-dir", type=Path, default=None)
    parser.add_argument("--smoothing-window", type=int, default=50)
    parser.add_argument("--print-history", action="store_true")
    args = parser.parse_args()

    result = run_benchmark(
        episodes=args.episodes,
        max_steps=args.max_steps,
        repeats=args.repeats,
        seed=args.seed,
        counter_cap=args.counter_cap,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        epsilon=args.epsilon,
    )

    if args.plot_output_dir is not None:
        result["plots"] = save_plots(
            result,
            output_dir=args.plot_output_dir,
            smoothing_window=args.smoothing_window,
        )

    text = json.dumps(result, indent=2)

    if args.json_output is not None:
        args.json_output.write_text(text + "\n", encoding="utf-8")

    if args.print_history:
        print(text)
    else:
        print(json.dumps(_without_history(result), indent=2))


if __name__ == "__main__":
    main()
