# ruff: noqa: D101,D103,E501
"""Anakin: stoa LetterWorld cross-product -> single-DQN, end-to-end JAX.

Self-contained JAX-Anakin single-DQN trainer (copied from the proven PuckWorld
template ``examples/rm/anakin_puckworld/train_stoix_anakin.py``), adapted to
consume the stoa-wrapped LetterWorld cross-product env and to (a) use a plain
single-DQN target, (b) bootstrap on true termination only (not truncation), and
(c) run periodic greedy eval on a fresh non-auto-reset vmapped env so episode
termination is observable and "solved" (RM terminal) can be measured.
"""

from __future__ import annotations

import os
import time
from functools import partial
from pathlib import Path
from typing import NamedTuple

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # demo default; use gpu/metal for real runs

import jax
import jax.numpy as jnp
import numpy as np
import optax
from stoa import AutoResetWrapper, RecordEpisodeMetrics
from stoa.core_wrappers.vmap import VmapWrapper
from stoa.core_wrappers.wrapper import AddRNGKey
from stoa.env_types import StepType

from examples.rm.letterworld_anakin.letterworld_stoa_env import (
    make_letterworld_cross_product,
)

# Shared hyperparameters (LetterWorld set; must match train_sb3.py after the spike).
GAMMA = 0.99
LR = 1e-3
ROLLOUT = 16
MAX_STEPS = 100
HIDDEN = (64, 64)
BATCH = 128
BUFFER = 50_000
TARGET_UPDATE = 500
LEARNING_STARTS = 1_000
EXPLORATION_FRACTION = 0.2
EPS_START, EPS_END = 1.0, 0.05
EVAL_EVERY = 2_000
EVAL_EPISODES = 20

NUM_ACTIONS = 4
OBS_SIZE = 7


def build_wrapped_env(num_envs: int):
    """Apply the Stoix core-wrapper chain to the LetterWorld cross-product."""
    env = make_letterworld_cross_product(max_steps=MAX_STEPS)
    env = AddRNGKey(env)
    env = RecordEpisodeMetrics(env)
    env = AutoResetWrapper(env, next_obs_in_extras=True)
    env = VmapWrapper(env)
    return env


class ReplayBuffer(NamedTuple):
    obs: jax.Array
    actions: jax.Array
    next_obs: jax.Array
    rewards: jax.Array
    dones: jax.Array
    pos: jax.Array
    size: jax.Array


def init_replay_buffer(buffer_size: int) -> ReplayBuffer:
    return ReplayBuffer(
        obs=jnp.zeros((buffer_size, OBS_SIZE), dtype=jnp.float32),
        actions=jnp.zeros((buffer_size,), dtype=jnp.int32),
        next_obs=jnp.zeros((buffer_size, OBS_SIZE), dtype=jnp.float32),
        rewards=jnp.zeros((buffer_size,), dtype=jnp.float32),
        dones=jnp.zeros((buffer_size,), dtype=jnp.float32),
        pos=jnp.asarray(0, dtype=jnp.int32),
        size=jnp.asarray(0, dtype=jnp.int32),
    )


def _add_rows(buffer: ReplayBuffer, rows: tuple[jax.Array, ...]) -> ReplayBuffer:
    obs, actions, next_obs, rewards, dones = rows
    n = obs.shape[0]
    capacity = buffer.obs.shape[0]
    idx = (buffer.pos + jnp.arange(n, dtype=jnp.int32)) % capacity
    size = jnp.minimum(buffer.size + n, capacity)
    return ReplayBuffer(
        obs=buffer.obs.at[idx].set(obs),
        actions=buffer.actions.at[idx].set(actions.astype(jnp.int32)),
        next_obs=buffer.next_obs.at[idx].set(next_obs),
        rewards=buffer.rewards.at[idx].set(rewards.astype(jnp.float32)),
        dones=buffer.dones.at[idx].set(dones.astype(jnp.float32)),
        pos=(buffer.pos + n) % capacity,
        size=size,
    )


def _sample_batch(buffer: ReplayBuffer, key: jax.Array, batch_size: int) -> tuple[jax.Array, ...]:
    idx = jax.random.randint(key, (batch_size,), 0, jnp.maximum(buffer.size, 1))
    return (
        buffer.obs[idx],
        buffer.actions[idx],
        buffer.next_obs[idx],
        buffer.rewards[idx],
        buffer.dones[idx],
    )


def init_mlp_params(
    key: jax.Array,
    *,
    input_dim: int = OBS_SIZE,
    hidden_sizes: tuple[int, ...] = HIDDEN,
    output_dim: int = NUM_ACTIONS,
) -> list[dict[str, jax.Array]]:
    dims = (input_dim, *hidden_sizes, output_dim)
    keys = jax.random.split(key, len(dims) - 1)
    params = []
    for layer_key, in_dim, out_dim in zip(keys, dims[:-1], dims[1:], strict=True):
        scale = jnp.sqrt(2.0 / float(in_dim))
        w = jax.random.normal(layer_key, (in_dim, out_dim), dtype=jnp.float32) * scale
        b = jnp.zeros((out_dim,), dtype=jnp.float32)
        params.append({"w": w, "b": b})
    return params


def mlp_apply(params: list[dict[str, jax.Array]], x: jax.Array) -> jax.Array:
    h = x
    for layer in params[:-1]:
        h = jax.nn.relu(h @ layer["w"] + layer["b"])
    return h @ params[-1]["w"] + params[-1]["b"]


def _epsilon_at_step(
    step: jax.Array,
    *,
    schedule_steps: int,
    eps_start: float = EPS_START,
    eps_end: float = EPS_END,
) -> jax.Array:
    progress = jnp.minimum(step.astype(jnp.float32) / float(max(schedule_steps, 1)), 1.0)
    return eps_start + progress * (eps_end - eps_start)


def _select_actions(
    params: list[dict[str, jax.Array]],
    obs: jax.Array,
    key: jax.Array,
    *,
    epsilon: jax.Array,
) -> jax.Array:
    q_values = mlp_apply(params, obs)
    greedy = jnp.argmax(q_values, axis=-1).astype(jnp.int32)
    action_key, explore_key = jax.random.split(key)
    random_actions = jax.random.randint(action_key, greedy.shape, 0, NUM_ACTIONS)
    explore = jax.random.uniform(explore_key, greedy.shape) < epsilon
    return jnp.where(explore, random_actions, greedy).astype(jnp.int32)


def _dqn_update(
    params: list[dict[str, jax.Array]],
    target_params: list[dict[str, jax.Array]],
    opt_state: optax.OptState,
    batch: tuple[jax.Array, ...],
    optimizer: optax.GradientTransformation,
) -> tuple[list[dict[str, jax.Array]], optax.OptState, dict[str, jax.Array]]:
    obs, actions, next_obs, rewards, dones = batch

    def loss_fn(current_params: list[dict[str, jax.Array]]) -> tuple[jax.Array, jax.Array]:
        q_values = mlp_apply(current_params, obs)
        action_q = jnp.take_along_axis(q_values, actions[:, None], axis=1)[:, 0]
        # Single-DQN target: plain max over the target network's next-state Q values.
        next_q = jnp.max(mlp_apply(target_params, next_obs), axis=-1)
        target = rewards + (1.0 - dones) * GAMMA * jax.lax.stop_gradient(next_q)
        loss = optax.huber_loss(action_q, target).mean()
        return loss, action_q.mean()

    (loss, q_mean), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, {"loss": loss, "q_mean": q_mean}


def _make_rollout_chunk_fn(env, *, num_envs: int, schedule_steps: int):
    """Build a jittable rollout-of-``ROLLOUT``-steps function closing over ``env``.

    ``env`` (a stoa ``Wrapper`` instance) is not a JAX pytree/hashable static
    value, so it cannot be passed as a jit argument; it is instead captured in
    a closure and the returned function is jitted once per ``run_anakin_dqn``
    call and reused (traced once, cached) across all outer iterations.
    """

    @jax.jit
    def rollout_chunk(
        params: list[dict[str, jax.Array]],
        env_state,
        ts,
        buffer: ReplayBuffer,
        key: jax.Array,
        env_step: jax.Array,
    ):
        def body(carry, _):
            env_state, ts, buffer, key, env_step = carry
            key, action_key = jax.random.split(key)
            epsilon = _epsilon_at_step(env_step, schedule_steps=schedule_steps)
            actions = _select_actions(params, ts.observation, action_key, epsilon=epsilon)
            obs_before = ts.observation
            env_state, ts = env.step(env_state, actions)
            # Bootstrap only on TRUE termination (RM terminal), never on truncation,
            # so a time-limit cut-off does not zero out the bootstrap value.
            is_terminated = ts.step_type == StepType.TERMINATED
            done = is_terminated.astype(jnp.float32)
            # With AutoResetWrapper(next_obs_in_extras=True), the true post-step
            # observation (pre-auto-reset) lives in extras["next_obs"]; ts.observation
            # is instead the reset observation whenever an episode ends.
            true_next_obs = ts.extras["next_obs"]
            rows = (obs_before, actions, true_next_obs, ts.reward, done)
            buffer = _add_rows(buffer, rows)
            env_step = env_step + num_envs
            episode_metrics = ts.extras["episode_metrics"]
            return (env_state, ts, buffer, key, env_step), (
                episode_metrics["episode_return"],
                episode_metrics["is_terminal_step"],
            )

        (env_state, ts, buffer, key, env_step), (ep_returns, ep_dones) = jax.lax.scan(
            body, (env_state, ts, buffer, key, env_step), None, length=ROLLOUT
        )
        return env_state, ts, buffer, key, env_step, ep_returns, ep_dones

    return rollout_chunk


def _make_update_chunk_fn(optimizer: optax.GradientTransformation):
    """Build a jittable single-DQN update-chunk function closing over ``optimizer``.

    ``optimizer`` is an ``optax.GradientTransformation`` (a NamedTuple of
    Python functions), not array data, so it cannot be passed as a jit
    argument; it is captured in a closure instead, and the returned function
    is jitted once per ``run_anakin_dqn`` call and reused across iterations.
    """

    @partial(jax.jit, static_argnames=("num_updates",))
    def update_chunk(
        params: list[dict[str, jax.Array]],
        target_params: list[dict[str, jax.Array]],
        opt_state: optax.OptState,
        buffer: ReplayBuffer,
        key: jax.Array,
        *,
        num_updates: int,
    ):
        """Run ``num_updates`` single-DQN gradient steps via a jittable ``lax.scan``."""

        def body(carry, _):
            params, opt_state, key = carry
            key, sample_key = jax.random.split(key)
            batch = _sample_batch(buffer, sample_key, BATCH)
            params, opt_state, info = _dqn_update(
                params, target_params, opt_state, batch, optimizer
            )
            return (params, opt_state, key), info

        (params, opt_state, key), infos = jax.lax.scan(
            body, (params, opt_state, key), None, length=num_updates
        )
        return (
            params,
            opt_state,
            key,
            {"loss": infos["loss"][-1], "q_mean": infos["q_mean"][-1]},
        )

    return update_chunk


def _make_eval_env():
    """Build a fresh, non-auto-reset vmapped LetterWorld env for greedy eval."""
    env = make_letterworld_cross_product(max_steps=MAX_STEPS)
    env = AddRNGKey(env)
    env = VmapWrapper(env)
    return env


def _greedy_eval(params, eval_env, key, num_episodes):
    """Return (success_rate, mean_return) over ``num_episodes`` greedy episodes."""
    keys = jax.random.split(key, num_episodes)
    state, ts = eval_env.reset(keys)
    ret = jnp.zeros(num_episodes)
    done = jnp.zeros(num_episodes, dtype=bool)
    success = jnp.zeros(num_episodes, dtype=bool)

    def body(carry, _):
        state, ts, ret, done, success = carry
        actions = jnp.argmax(mlp_apply(params, ts.observation), axis=-1).astype(jnp.int32)
        state, ts = eval_env.step(state, actions)
        active = jnp.logical_not(done)
        ret = ret + jnp.where(active, ts.reward, 0.0)
        term = jnp.logical_and(active, ts.step_type == StepType.TERMINATED)
        success = jnp.logical_or(success, term)
        done = jnp.logical_or(done, ts.step_type != StepType.MID)
        return (state, ts, ret, done, success), None

    (state, ts, ret, done, success), _ = jax.lax.scan(
        body, (state, ts, ret, done, success), None, length=MAX_STEPS
    )
    return success.mean(), ret.mean()


def run_anakin_dqn(
    total_timesteps: int, seed: int, log_dir: str, num_envs: int = 64
) -> dict:
    """Train a single-DQN end-to-end in JAX on the wrapped LetterWorld env.

    Anakin-style: vmapped envs, a jittable rollout of ``ROLLOUT`` steps per
    iteration (collected via ``lax.scan``), and a jitted single-DQN update
    sampling minibatches of ``BATCH`` from an in-memory replay buffer. Periodic
    greedy eval (every ``EVAL_EVERY`` env-steps over ``EVAL_EPISODES`` episodes)
    records the eval curve; "solved" is an eval reaching the RM terminal
    (``StepType.TERMINATED``) with a success rate of at least 0.95.
    """
    env = build_wrapped_env(num_envs)
    optimizer = optax.adam(LR)

    key = jax.random.PRNGKey(seed)
    key, params_key, reset_key, eval_key = jax.random.split(key, 4)
    params = init_mlp_params(params_key)
    target_params = jax.tree_util.tree_map(lambda x: x.copy(), params)
    opt_state = optimizer.init(params)
    buffer = init_replay_buffer(BUFFER)

    schedule_steps = max(int(EXPLORATION_FRACTION * total_timesteps), 1)
    n_iterations = max(total_timesteps // (num_envs * ROLLOUT), 1)
    updates_per_rollout = max((num_envs * ROLLOUT) // 4, 1)
    target_update_every = max(TARGET_UPDATE // (num_envs * ROLLOUT), 1)

    reset_keys = jax.random.split(reset_key, num_envs)
    env_state, ts = jax.jit(env.reset)(reset_keys)

    rollout_chunk = _make_rollout_chunk_fn(
        env, num_envs=num_envs, schedule_steps=schedule_steps
    )
    update_chunk = _make_update_chunk_fn(optimizer)

    eval_env = _make_eval_env()
    eval_fn = jax.jit(
        partial(_greedy_eval, eval_env=eval_env, num_episodes=EVAL_EPISODES)
    )

    eval_steps: list[int] = []
    eval_wall: list[float] = []
    success_rate: list[float] = []
    mean_return: list[float] = []
    time_to_solve: float | None = None
    steps_to_solve: int | None = None

    def _run_eval(env_step_host: int) -> None:
        nonlocal eval_key, time_to_solve, steps_to_solve
        eval_key, sub = jax.random.split(eval_key)
        sr, mr = eval_fn(params, key=sub)
        sr_host = float(np.asarray(sr))
        mr_host = float(np.asarray(mr))
        wall = time.perf_counter() - start
        eval_steps.append(int(env_step_host))
        eval_wall.append(wall)
        success_rate.append(sr_host)
        mean_return.append(mr_host)
        if sr_host >= 0.95 and time_to_solve is None:
            time_to_solve = wall
            steps_to_solve = int(env_step_host)

    start = time.perf_counter()
    env_step = jnp.asarray(0, dtype=jnp.int32)
    env_step_host = 0
    next_eval_at = EVAL_EVERY

    # Baseline eval at step 0 (untrained greedy policy).
    _run_eval(0)

    for iteration in range(n_iterations):
        env_state, ts, buffer, key, env_step, ep_returns, ep_dones = rollout_chunk(
            params,
            env_state,
            ts,
            buffer,
            key,
            env_step,
        )
        env_step_host += num_envs * ROLLOUT

        buffer_size_host = int(np.asarray(buffer.size))
        if buffer_size_host >= max(LEARNING_STARTS, BATCH):
            params, opt_state, key, _info = update_chunk(
                params,
                target_params,
                opt_state,
                buffer,
                key,
                num_updates=updates_per_rollout,
            )

        if (iteration + 1) % target_update_every == 0:
            target_params = jax.tree_util.tree_map(lambda x: x.copy(), params)

        if env_step_host >= next_eval_at:
            _run_eval(env_step_host)
            next_eval_at += EVAL_EVERY

    # Ensure a final eval reflects the end-of-training policy.
    if not eval_steps or eval_steps[-1] != env_step_host:
        _run_eval(env_step_host)

    elapsed = time.perf_counter() - start

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(log_dir) / f"anakin_dqn_seed{seed}_progress.csv"
    with csv_path.open("w") as fh:
        fh.write("step,wall_clock,success_rate,mean_return\n")
        for s, w, sr, mr in zip(
            eval_steps, eval_wall, success_rate, mean_return, strict=True
        ):
            fh.write(f"{s},{w:.4f},{sr:.4f},{mr:.4f}\n")

    return {
        "backend": "anakin_dqn",
        "eval_steps": eval_steps,
        "eval_wall": eval_wall,
        "success_rate": success_rate,
        "mean_return": mean_return,
        "steps_per_sec": env_step_host / max(elapsed, 1e-9),
        "total_wall_clock": elapsed,
        "time_to_solve": time_to_solve,
        "steps_to_solve": steps_to_solve,
    }


if __name__ == "__main__":
    metrics = run_anakin_dqn(total_timesteps=150_000, seed=0, log_dir="results")
    print(
        f"Anakin DQN: {metrics['steps_per_sec']:.0f} steps/s, "
        f"{metrics['total_wall_clock']:.1f}s, "
        f"best success {max(metrics['success_rate']):.2f}, "
        f"solved@ {metrics['time_to_solve']} {metrics['steps_to_solve']}"
    )
