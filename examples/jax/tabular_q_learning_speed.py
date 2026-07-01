import argparse
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from examples.crm.tabular.core.crossproduct import (
    OfficeWorldCrossProduct as CrmOfficeWorldCrossProduct,
)
from examples.crm.tabular.core.ground import OfficeWorld as CrmOfficeWorld
from examples.crm.tabular.core.label import (
    OfficeWorldLabellingFunction as CrmOfficeWorldLabellingFunction,
)
from examples.crm.tabular.core.machine import OfficeWorldCountingRewardMachine
from examples.introduction.core.crossproduct import LetterWorldCrossProduct
from examples.introduction.core.ground import LetterWorld
from examples.introduction.core.label import LetterWorldLabellingFunction
from examples.introduction.core.machine import LetterWorldCountingRewardMachine
from examples.rm.tabular.core.crossproduct import (
    OfficeWorldCrossProduct as RmOfficeWorldCrossProduct,
)
from examples.rm.tabular.core.ground import OfficeWorld as RmOfficeWorld
from examples.rm.tabular.core.label import (
    OfficeWorldLabellingFunction as RmOfficeWorldLabellingFunction,
)
from examples.rm.tabular.core.machine import OfficeWorldRewardMachine
from pycrm.automaton import CountingRewardMachine, RmToCrmAdapter
from pycrm.crossproduct import CrossProduct
from pycrm.jax import FunctionalJaxCrossProduct, compile_crm

N_ACTIONS = 4

LETTER_SYMBOL_STATES = 2
LETTER_N_ROWS = LetterWorld.N_ROWS
LETTER_N_COLS = LetterWorld.N_COLS

OFFICE_N_ROWS = 13
OFFICE_N_COLS = 17
OFFICE_MAIL_STATES = 2
OFFICE_MAX_MAIL = 2


@dataclass(frozen=True)
class TabularBenchmarkTask:
    """One tabular, non-function-approximation benchmark target."""

    name: str
    title: str
    make_python_env: Callable[[int], CrossProduct]
    make_jax_env: Callable[[int], FunctionalJaxCrossProduct]
    q_shape: Callable[[int], tuple[int, ...]]
    python_obs_index: Callable[[np.ndarray, int], tuple[int, ...]]
    jax_obs_index: Callable[[Any, int], tuple[Any, ...]]
    jax_obs_batch_index: Callable[[Any, int], tuple[Any, ...]]
    default_counter_cap: int
    n_actions: int = N_ACTIONS


def run_suite_benchmark(
    *,
    task_names: Sequence[str] | None = None,
    episodes: int = 5000,
    max_steps: int = 500,
    repeats: int = 3,
    seed: int = 0,
    counter_cap: int | None = None,
    learning_rate: float = 0.1,
    discount_factor: float = 0.99,
    epsilon: float = 0.1,
) -> dict[str, Any]:
    """Benchmark all selected tabular demos with Python and JAX training loops."""
    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    if repeats <= 0:
        raise ValueError("repeats must be positive.")
    if counter_cap is not None and counter_cap < 0:
        raise ValueError("counter_cap must be non-negative.")

    tasks = _select_tasks(task_names)
    task_results = {}

    for task_idx, task in enumerate(tasks):
        task_counter_cap = (
            task.default_counter_cap if counter_cap is None else counter_cap
        )
        task_results[task.name] = run_task_benchmark(
            task=task,
            episodes=episodes,
            max_steps=max_steps,
            repeats=repeats,
            seed=seed + 100_000 * task_idx,
            counter_cap=task_counter_cap,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
        )

    return {
        "config": {
            "tasks": [task.name for task in tasks],
            "episodes": episodes,
            "max_steps": max_steps,
            "repeats": repeats,
            "seed": seed,
            "counter_cap": counter_cap,
            "learning_rate": learning_rate,
            "discount_factor": discount_factor,
            "epsilon": epsilon,
        },
        "tasks": task_results,
        "note": (
            "JAX compile/first-call time is reported separately from "
            "steady-state time."
        ),
    }


def run_task_benchmark(
    *,
    task: TabularBenchmarkTask,
    episodes: int,
    max_steps: int,
    repeats: int,
    seed: int,
    counter_cap: int,
    learning_rate: float,
    discount_factor: float,
    epsilon: float,
) -> dict[str, Any]:
    """Run Python and JAX Q-learning benchmarks for one tabular task."""
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
        task=task,
        repeats=repeats,
        seed=seed,
        **training_config,
    )
    python_cql_times, python_cql_returns = _time_python_training(
        _run_python_counterfactual_q_learning,
        task=task,
        repeats=repeats,
        seed=seed + 1_000,
        **training_config,
    )

    train_jit = _make_jax_training_fn(
        task=task,
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
        task=task,
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
        "title": task.title,
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
        },
    }


def save_plots(
    result: dict[str, Any], output_dir: Path, smoothing_window: int = 50
) -> dict[str, str]:
    """Save suite returns, time, and speedup plots."""
    if smoothing_window <= 0:
        raise ValueError("smoothing_window must be positive.")

    output_dir.mkdir(parents=True, exist_ok=True)
    returns_path = output_dir / "tabular_returns.png"
    times_path = output_dir / "tabular_times.png"
    speedups_path = output_dir / "tabular_speedups.png"

    _plot_suite_returns(result, returns_path, smoothing_window)
    _plot_suite_times(result, times_path)
    _plot_suite_speedups(result, speedups_path)

    return {
        "returns_curve": str(returns_path),
        "times": str(times_path),
        "speedups": str(speedups_path),
    }


def _time_python_training(
    training_fn,
    *,
    task: TabularBenchmarkTask,
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
            task=task,
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
    task: TabularBenchmarkTask,
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

    cross_product = task.make_python_env(max_steps)
    q_table = np.zeros(task.q_shape(counter_cap), dtype=np.float32)
    returns = np.zeros(episodes, dtype=np.float32)

    for episode in range(episodes):
        obs, _ = cross_product.reset()
        done = False
        episode_return = 0.0

        while not done:
            state_idx = task.python_obs_index(obs, counter_cap)
            q_values = q_table[state_idx]
            if rng.random() < epsilon or np.all(q_values == 0):
                action = int(rng.integers(task.n_actions))
            else:
                action = int(np.argmax(q_values))

            next_obs, reward, terminated, truncated, _ = cross_product.step(action)
            done = terminated or truncated
            episode_return += float(reward)

            q_table = _python_update(
                q_table=q_table,
                state_idx=state_idx,
                action=action,
                next_state_idx=task.python_obs_index(next_obs, counter_cap),
                reward=float(reward),
                done=done,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
            )
            obs = next_obs

        returns[episode] = episode_return

    return returns


def _run_python_counterfactual_q_learning(
    *,
    task: TabularBenchmarkTask,
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

    cross_product = task.make_python_env(max_steps)
    q_table = np.zeros(task.q_shape(counter_cap), dtype=np.float32)
    returns = np.zeros(episodes, dtype=np.float32)

    for episode in range(episodes):
        obs, _ = cross_product.reset()
        done = False
        episode_return = 0.0

        while not done:
            state_idx = task.python_obs_index(obs, counter_cap)
            q_values = q_table[state_idx]
            if rng.random() < epsilon or np.all(q_values == 0):
                action = int(rng.integers(task.n_actions))
            else:
                action = int(np.argmax(q_values))

            next_obs, reward, terminated, truncated, _ = cross_product.step(action)
            done = terminated or truncated
            episode_return += float(reward)

            for cf_obs, cf_action, cf_next_obs, cf_reward, cf_done, _ in zip(
                *cross_product.generate_counterfactual_experience(
                    cross_product.to_ground_obs(obs),
                    action,
                    cross_product.to_ground_obs(next_obs),
                ),
                strict=True,
            ):
                q_table = _python_update(
                    q_table=q_table,
                    state_idx=task.python_obs_index(cf_obs, counter_cap),
                    action=int(cf_action),
                    next_state_idx=task.python_obs_index(cf_next_obs, counter_cap),
                    reward=float(cf_reward),
                    done=bool(cf_done),
                    learning_rate=learning_rate,
                    discount_factor=discount_factor,
                )

            q_table = _python_update(
                q_table=q_table,
                state_idx=state_idx,
                action=action,
                next_state_idx=task.python_obs_index(next_obs, counter_cap),
                reward=float(reward),
                done=done,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
            )
            obs = next_obs

        returns[episode] = episode_return

    return returns


def _python_update(
    *,
    q_table: np.ndarray,
    state_idx: tuple[int, ...],
    action: int,
    next_state_idx: tuple[int, ...],
    reward: float,
    done: bool,
    learning_rate: float,
    discount_factor: float,
) -> np.ndarray:
    td_target = reward
    if not done:
        td_target += discount_factor * float(np.max(q_table[next_state_idx]))

    old_value = q_table[state_idx + (action,)]
    q_table[state_idx + (action,)] = old_value + learning_rate * (
        td_target - old_value
    )
    return q_table


def _make_jax_training_fn(
    *,
    task: TabularBenchmarkTask,
    counterfactual: bool,
    episodes: int,
    max_steps: int,
    counter_cap: int,
    learning_rate: float,
    discount_factor: float,
    epsilon: float,
):
    env = task.make_jax_env(max_steps)
    q_shape = task.q_shape(counter_cap)

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
                    state_idx = task.jax_obs_index(obs, counter_cap)
                    q_values = q_table[state_idx]
                    explore = jax.random.uniform(eps_key) < epsilon
                    unvisited = jnp.all(q_values == 0)
                    random_action = jax.random.randint(
                        action_key, (), 0, task.n_actions, dtype=jnp.int32
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
                            task=task,
                            q_table=q_table,
                            ground_obs=state.ground_obs,
                            action=action,
                            next_ground_obs=next_state.ground_obs,
                            counter_cap=counter_cap,
                            learning_rate=learning_rate,
                            discount_factor=discount_factor,
                        )

                    next_state_idx = task.jax_obs_index(next_obs, counter_cap)
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
    env: FunctionalJaxCrossProduct,
    task: TabularBenchmarkTask,
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

    cf_state_idx = task.jax_obs_batch_index(cf_obs, counter_cap)
    cf_next_state_idx = task.jax_obs_batch_index(cf_next_obs, counter_cap)
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


def _make_letter_python_env(max_steps: int) -> LetterWorldCrossProduct:
    return LetterWorldCrossProduct(
        ground_env=LetterWorld(),
        crm=LetterWorldCountingRewardMachine(),
        lf=LetterWorldLabellingFunction(),
        max_steps=max_steps,
    )


def _make_letter_jax_env(max_steps: int) -> FunctionalJaxCrossProduct:
    return FunctionalJaxCrossProduct(
        compiled_crm=compile_crm(LetterWorldCountingRewardMachine()),
        reset_fn=_jax_letter_reset,
        step_fn=_jax_letter_step,
        label_fn=_jax_letter_label,
        max_steps=max_steps,
        obs_fn=_jax_letter_obs,
    )


def _jax_letter_reset(key):
    del key
    return jnp.asarray([0, 1, 3], dtype=jnp.int32)


def _jax_letter_step(obs, action, key):
    symbol_seen, row, col = obs

    row_next = jnp.select(
        [action == LetterWorld.UP, action == LetterWorld.DOWN],
        [jnp.maximum(row - 1, 0), jnp.minimum(row + 1, LETTER_N_ROWS - 1)],
        default=row,
    )
    col_next = jnp.select(
        [action == LetterWorld.RIGHT, action == LetterWorld.LEFT],
        [jnp.minimum(col + 1, LETTER_N_COLS - 1), jnp.maximum(col - 1, 0)],
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


def _letter_q_shape(counter_cap: int) -> tuple[int, int, int, int, int, int]:
    crm = LetterWorldCountingRewardMachine()
    return (
        LETTER_SYMBOL_STATES,
        LETTER_N_ROWS,
        LETTER_N_COLS,
        _num_machine_states(crm),
        counter_cap + 1,
        N_ACTIONS,
    )


def _letter_python_obs_index(
    obs: np.ndarray, counter_cap: int
) -> tuple[int, int, int, int, int]:
    symbol_seen, row, col, u, c = [int(value) for value in obs]
    return symbol_seen, row, col, u, min(max(c, 0), counter_cap)


def _letter_jax_obs_index(obs, counter_cap: int):
    c = jnp.clip(obs[4], 0, counter_cap)
    return obs[0], obs[1], obs[2], obs[3], c


def _letter_jax_obs_batch_index(obs, counter_cap: int):
    c = jnp.clip(obs[:, 4], 0, counter_cap)
    return obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3], c


def _make_office_rm_python_env(max_steps: int) -> RmOfficeWorldCrossProduct:
    return RmOfficeWorldCrossProduct(
        ground_env=RmOfficeWorld(),
        machine=OfficeWorldRewardMachine(),
        lf=RmOfficeWorldLabellingFunction(),
        max_steps=max_steps,
    )


def _make_office_rm_jax_env(max_steps: int) -> FunctionalJaxCrossProduct:
    return FunctionalJaxCrossProduct(
        compiled_crm=compile_crm(OfficeWorldRewardMachine()),
        reset_fn=_jax_office_reset,
        step_fn=_jax_office_step,
        label_fn=_jax_office_label,
        max_steps=max_steps,
        obs_fn=_jax_office_rm_obs,
    )


def _office_rm_q_shape(counter_cap: int) -> tuple[int, int, int, int, int]:
    del counter_cap
    crm = RmToCrmAdapter(OfficeWorldRewardMachine())
    return (
        OFFICE_N_ROWS,
        OFFICE_N_COLS,
        OFFICE_MAIL_STATES,
        _num_machine_states(crm),
        N_ACTIONS,
    )


def _office_rm_python_obs_index(
    obs: np.ndarray, counter_cap: int
) -> tuple[int, int, int, int]:
    del counter_cap
    row, col, mail_empty, u = [int(value) for value in obs]
    return row, col, mail_empty, u


def _office_rm_jax_obs_index(obs, counter_cap: int):
    del counter_cap
    return obs[0], obs[1], obs[2], obs[3]


def _office_rm_jax_obs_batch_index(obs, counter_cap: int):
    del counter_cap
    return obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]


def _make_office_crm_python_env(max_steps: int) -> CrmOfficeWorldCrossProduct:
    return CrmOfficeWorldCrossProduct(
        ground_env=CrmOfficeWorld(),
        crm=OfficeWorldCountingRewardMachine(),
        lf=CrmOfficeWorldLabellingFunction(),
        max_steps=max_steps,
    )


def _make_office_crm_jax_env(max_steps: int) -> FunctionalJaxCrossProduct:
    return FunctionalJaxCrossProduct(
        compiled_crm=compile_crm(OfficeWorldCountingRewardMachine()),
        reset_fn=_jax_office_reset,
        step_fn=_jax_office_step,
        label_fn=_jax_office_label,
        max_steps=max_steps,
        obs_fn=_jax_office_crm_obs,
    )


def _office_crm_q_shape(counter_cap: int) -> tuple[int, int, int, int, int, int, int]:
    crm = OfficeWorldCountingRewardMachine()
    return (
        OFFICE_N_ROWS,
        OFFICE_N_COLS,
        OFFICE_MAIL_STATES,
        _num_machine_states(crm),
        counter_cap + 1,
        counter_cap + 1,
        N_ACTIONS,
    )


def _office_crm_python_obs_index(
    obs: np.ndarray, counter_cap: int
) -> tuple[int, int, int, int, int, int]:
    row, col, mail_empty, u, c0, c1 = [int(value) for value in obs]
    return (
        row,
        col,
        mail_empty,
        u,
        min(max(c0, 0), counter_cap),
        min(max(c1, 0), counter_cap),
    )


def _office_crm_jax_obs_index(obs, counter_cap: int):
    c0 = jnp.clip(obs[4], 0, counter_cap)
    c1 = jnp.clip(obs[5], 0, counter_cap)
    return obs[0], obs[1], obs[2], obs[3], c0, c1


def _office_crm_jax_obs_batch_index(obs, counter_cap: int):
    c0 = jnp.clip(obs[:, 4], 0, counter_cap)
    c1 = jnp.clip(obs[:, 5], 0, counter_cap)
    return obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3], c0, c1


def _jax_office_reset(key):
    total_mail = jax.random.randint(
        key, (), 1, OFFICE_MAX_MAIL + 1, dtype=jnp.int32
    )
    return jnp.asarray([3, 11, 0, 0, total_mail], dtype=jnp.int32)


def _jax_office_step(obs, action, key):
    del key
    row, col, mail_empty, mail_collected, total_mail = obs

    candidate_row = jnp.select(
        [action == RmOfficeWorld.UP, action == RmOfficeWorld.DOWN],
        [row - 1, row + 1],
        default=row,
    )
    candidate_col = jnp.select(
        [action == RmOfficeWorld.RIGHT, action == RmOfficeWorld.LEFT],
        [col + 1, col - 1],
        default=col,
    )
    blocked = OFFICE_WALLS[candidate_row, candidate_col]
    row_next = jnp.where(blocked, row, candidate_row)
    col_next = jnp.where(blocked, col, candidate_col)

    at_mail = (row_next == 6) & (col_next == 10)
    should_collect = at_mail & (mail_collected < total_mail)
    mail_collected_next = jnp.where(
        should_collect, mail_collected + 1, mail_collected
    )
    mail_empty_next = jnp.where(
        at_mail & (~should_collect), jnp.asarray(1, dtype=jnp.int32), mail_empty
    )

    return jnp.asarray(
        [
            row_next,
            col_next,
            mail_empty_next,
            mail_collected_next,
            total_mail,
        ],
        dtype=jnp.int32,
    )


def _jax_office_label(obs, action, next_obs):
    del obs, action
    row, col, mail_empty = next_obs[:3]
    at_coffee = (row == 2) & (col == 6)
    at_mail = (row == 6) & (col == 10)
    at_people = (row == 6) & (col == 6)
    at_decoration = (
        ((row == 6) & (col == 2))
        | ((row == 6) & (col == 14))
        | ((row == 10) & (col == 6))
        | ((row == 10) & (col == 10))
    )
    return jnp.asarray(
        [
            at_coffee,
            at_mail & (mail_empty == 0),
            at_mail & (mail_empty == 1),
            at_people,
            at_decoration,
        ],
        dtype=jnp.bool_,
    )


def _jax_office_rm_obs(ground_obs, u, c):
    del c
    return jnp.asarray(
        [ground_obs[0], ground_obs[1], ground_obs[2], u], dtype=jnp.int32
    )


def _jax_office_crm_obs(ground_obs, u, c):
    return jnp.asarray(
        [ground_obs[0], ground_obs[1], ground_obs[2], u, c[0], c[1]],
        dtype=jnp.int32,
    )


def _num_machine_states(crm: CountingRewardMachine) -> int:
    return max([crm.u_0, *crm.U, *crm.F]) + 1


def _office_wall_grid() -> jax.Array:
    grid = np.zeros((OFFICE_N_ROWS, OFFICE_N_COLS), dtype=np.bool_)
    grid[0, :] = True
    grid[12, :] = True
    grid[:, 0] = True
    grid[:, 16] = True
    grid[4, 1] = True
    grid[4, 15] = True
    grid[4, 3:6] = True
    grid[4, 7:10] = True
    grid[4, 11:14] = True
    grid[8, 1] = True
    grid[8, 15] = True
    grid[8, 3:14] = True

    for row in range(1, 12):
        if row in (2, 10):
            continue
        for col in range(OFFICE_N_COLS):
            if col in (4, 8, 12):
                grid[row, col] = True
    return jnp.asarray(grid)


OFFICE_WALLS = _office_wall_grid()


TASKS: tuple[TabularBenchmarkTask, ...] = (
    TabularBenchmarkTask(
        name="letter",
        title="Letter World CRM",
        make_python_env=_make_letter_python_env,
        make_jax_env=_make_letter_jax_env,
        q_shape=_letter_q_shape,
        python_obs_index=_letter_python_obs_index,
        jax_obs_index=_letter_jax_obs_index,
        jax_obs_batch_index=_letter_jax_obs_batch_index,
        default_counter_cap=12,
    ),
    TabularBenchmarkTask(
        name="office-rm",
        title="OfficeWorld RM",
        make_python_env=_make_office_rm_python_env,
        make_jax_env=_make_office_rm_jax_env,
        q_shape=_office_rm_q_shape,
        python_obs_index=_office_rm_python_obs_index,
        jax_obs_index=_office_rm_jax_obs_index,
        jax_obs_batch_index=_office_rm_jax_obs_batch_index,
        default_counter_cap=0,
    ),
    TabularBenchmarkTask(
        name="office-crm",
        title="OfficeWorld CRM",
        make_python_env=_make_office_crm_python_env,
        make_jax_env=_make_office_crm_jax_env,
        q_shape=_office_crm_q_shape,
        python_obs_index=_office_crm_python_obs_index,
        jax_obs_index=_office_crm_jax_obs_index,
        jax_obs_batch_index=_office_crm_jax_obs_batch_index,
        default_counter_cap=3,
    ),
)
TASKS_BY_NAME = {task.name: task for task in TASKS}


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


def _return_summary(returns: np.ndarray) -> dict[str, float]:
    recent_window = min(50, len(returns))
    return {
        "final_return": float(returns[-1]),
        "recent_average_return": float(np.mean(returns[-recent_window:])),
    }


def _plot_suite_returns(
    result: dict[str, Any], output_path: Path, smoothing_window: int
) -> None:
    import matplotlib.pyplot as plt

    task_items = list(result["tasks"].items())
    fig, axes = plt.subplots(
        len(task_items),
        1,
        figsize=(11, max(4.0, 3.5 * len(task_items))),
        squeeze=False,
        sharex=False,
    )
    colors = {
        "python_returns": "#2f6fbb",
        "python_counterfactual_returns": "#4f7f45",
        "jax_returns": "#c95f2d",
        "jax_counterfactual_returns": "#7b4fa3",
    }
    labels = {
        "python_returns": "Python Q-learning",
        "python_counterfactual_returns": "Python CQL",
        "jax_returns": "JAX Q-learning",
        "jax_counterfactual_returns": "JAX CQL",
    }

    for ax, (_, task_result) in zip(axes[:, 0], task_items, strict=True):
        run_times = {
            "python_returns": task_result["python"]["times_s"],
            "python_counterfactual_returns": task_result["python_counterfactual"][
                "times_s"
            ],
            "jax_returns": task_result["jax"]["steady_times_s"],
            "jax_counterfactual_returns": task_result["jax_counterfactual"][
                "steady_times_s"
            ],
        }
        for history_key, color in colors.items():
            _plot_return_family(
                ax=ax,
                returns=np.asarray(
                    task_result["history"][history_key], dtype=np.float32
                ),
                run_times=np.asarray(run_times[history_key], dtype=np.float32),
                label=labels[history_key],
                color=color,
                smoothing_window=smoothing_window,
            )
        ax.set_title(task_result["title"])
        ax.set_xlabel("Training time (seconds; JAX compile excluded)")
        ax.set_ylabel("Episode return")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_return_family(
    *,
    ax,
    returns: np.ndarray,
    run_times: np.ndarray,
    label: str,
    color: str,
    smoothing_window: int,
) -> None:
    smoothed_rows = []
    time_rows = []

    for row, run_time in zip(returns, run_times, strict=True):
        smoothed = _moving_average(row, smoothing_window)
        smoothed_rows.append(smoothed)
        time_rows.append(
            _elapsed_time_axis(
                total_time_s=float(run_time),
                episodes=len(row),
                points=len(smoothed),
            )
        )

    smoothed = np.asarray(smoothed_rows, dtype=np.float32)
    elapsed_times = np.asarray(time_rows, dtype=np.float32)

    if len(smoothed) > 1:
        for time_row, return_row in zip(elapsed_times, smoothed, strict=True):
            ax.plot(time_row, return_row, color=color, alpha=0.18, linewidth=1)

    ax.plot(
        np.mean(elapsed_times, axis=0),
        np.mean(smoothed, axis=0),
        color=color,
        linewidth=2.4,
        label=label,
    )


def _plot_suite_times(result: dict[str, Any], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = []
    python_means = []
    jax_steady_means = []
    jax_compile_first = []

    for task_result in result["tasks"].values():
        labels.extend([f"{task_result['title']}\nQ", f"{task_result['title']}\nCQL"])
        python_means.extend(
            [
                task_result["python"]["mean_s"],
                task_result["python_counterfactual"]["mean_s"],
            ]
        )
        jax_steady_means.extend(
            [
                task_result["jax"]["steady_mean_s"],
                task_result["jax_counterfactual"]["steady_mean_s"],
            ]
        )
        jax_compile_first.extend(
            [
                task_result["jax"]["compile_first_call_s"],
                task_result["jax_counterfactual"]["compile_first_call_s"],
            ]
        )

    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.45), 5.8))
    ax.bar(x - width, python_means, width, label="Python", color="#2f6fbb")
    ax.bar(x, jax_steady_means, width, label="JAX steady", color="#c95f2d")
    ax.bar(
        x + width,
        jax_compile_first,
        width,
        label="JAX compile+run",
        color="#767676",
    )
    ax.set_title("Tabular Training Time")
    ax.set_ylabel("Seconds")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_suite_speedups(result: dict[str, Any], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = []
    python_means = []
    jax_steady_means = []
    speedups = []

    for task_result in result["tasks"].values():
        labels.extend([f"{task_result['title']}\nQ", f"{task_result['title']}\nCQL"])
        python_means.extend(
            [
                task_result["python"]["mean_s"],
                task_result["python_counterfactual"]["mean_s"],
            ]
        )
        jax_steady_means.extend(
            [
                task_result["jax"]["steady_mean_s"],
                task_result["jax_counterfactual"]["steady_mean_s"],
            ]
        )
        speedups.extend(
            [
                task_result["speedup"]["q_learning_steady_state_ratio"],
                task_result["speedup"]["counterfactual_steady_state_ratio"],
            ]
        )

    x = np.arange(len(labels))
    width = 0.34
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(max(13, len(labels) * 2.1), 5.6),
        gridspec_kw={"width_ratios": [1.45, 1.0]},
    )

    axes[0].bar(
        x - width / 2,
        python_means,
        width,
        label="Normal Python",
        color="#2f6fbb",
    )
    axes[0].bar(
        x + width / 2,
        jax_steady_means,
        width,
        label="JAX steady",
        color="#c95f2d",
    )
    for idx, speedup in enumerate(speedups):
        y = max(python_means[idx], jax_steady_means[idx])
        axes[0].annotate(
            f"{speedup:.2f}x",
            xy=(idx, y),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[0].set_title("Normal Python vs JAX Time")
    axes[0].set_ylabel("Seconds")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=24, ha="right")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)
    axes[0].legend()

    ratio_colors = ["#4f7f45" if "\nQ" in label else "#7b4fa3" for label in labels]
    axes[1].bar(x, speedups, width=0.58, color=ratio_colors)
    axes[1].axhline(1.0, color="#444444", linewidth=1.2, linestyle="--")
    axes[1].set_title("Steady-State Speedup")
    axes[1].set_ylabel("Python time / JAX time")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=24, ha="right")
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode="valid")


def _elapsed_time_axis(
    *, total_time_s: float, episodes: int, points: int
) -> np.ndarray:
    offset = episodes - points
    episode_numbers = np.arange(offset + 1, episodes + 1, dtype=np.float32)
    return episode_numbers / float(episodes) * total_time_s


def _without_history(result: dict[str, Any]) -> dict[str, Any]:
    stripped = {key: value for key, value in result.items() if key != "tasks"}
    stripped["tasks"] = {
        task_name: {
            key: value
            for key, value in task_result.items()
            if key != "history"
        }
        for task_name, task_result in result["tasks"].items()
    }
    return stripped


def main() -> None:
    """Run the command-line benchmark suite."""
    parser = argparse.ArgumentParser(
        description="Benchmark Python and JAX tabular Q-learning examples."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["all"],
        choices=["all", *TASKS_BY_NAME.keys()],
        help="Subset of tabular tasks to run.",
    )
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--counter-cap",
        type=int,
        default=None,
        help="Override the per-task counter clipping cap used by dense Q-tables.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--plot-output-dir", type=Path, default=None)
    parser.add_argument("--smoothing-window", type=int, default=50)
    parser.add_argument("--print-history", action="store_true")
    args = parser.parse_args()

    result = run_suite_benchmark(
        task_names=args.tasks,
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
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")

    if args.print_history:
        print(text)
    else:
        print(json.dumps(_without_history(result), indent=2))


if __name__ == "__main__":
    main()
