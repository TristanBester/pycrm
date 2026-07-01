# ruff: noqa: D101,D103,E501
"""Anakin: stoa PuckWorld cross-product -> DQN, end-to-end JAX.

Self-contained JAX-Anakin DQN fallback (Stoix is not installed). Ported from
``examples/rm/discrete-fast/jax_dqn.py`` (``run_jax_dqn`` + its ``ReplayBuffer``
NamedTuple, MLP, epsilon-greedy actor, optax update), adapted to consume the
stoa-wrapped PuckWorld cross-product env (B2) via the exact core-wrapper chain
proven to survive ``lax.scan`` in Phase A Task A4.
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

from examples.rm.anakin_puckworld.puckworld_stoa_env import make_puckworld_cross_product

# Shared hyperparameters (must match examples/rm/anakin_puckworld/train_sb3.py).
HIDDEN = (256, 256)
GAMMA = 0.99
LR = 2.5e-4
BATCH = 128
TARGET_UPDATE = 1_000
EXPLORATION_FRACTION = 0.2
MAX_STEPS = 1_000

ROLLOUT = 16
BUFFER_SIZE = 100_000
NUM_ACTIONS = 4
OBS_SIZE = 17


def build_wrapped_env(num_envs: int):
    """Apply the Stoix core-wrapper chain to the PuckWorld cross-product."""
    env = make_puckworld_cross_product(max_steps=MAX_STEPS)
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
    eps_start: float = 1.0,
    eps_end: float = 0.05,
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


def _ddqn_update(
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
        # Double DQN target: online net selects the next action, target net evaluates it.
        next_online_q = mlp_apply(current_params, next_obs)
        next_actions = jnp.argmax(next_online_q, axis=-1)
        next_target_q = mlp_apply(target_params, next_obs)
        next_q = jnp.take_along_axis(next_target_q, next_actions[:, None], axis=1)[:, 0]
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
            done = ts.last()
            # With AutoResetWrapper(next_obs_in_extras=True), the true post-step
            # observation (pre-auto-reset) lives in extras["next_obs"]; ts.observation
            # is instead the reset observation whenever an episode ends.
            true_next_obs = ts.extras["next_obs"]
            rows = (obs_before, actions, true_next_obs, ts.reward, done.astype(jnp.float32))
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
    """Build a jittable DDQN update-chunk function closing over ``optimizer``.

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
        """Run ``num_updates`` DDQN gradient steps via a jittable ``lax.scan``."""

        def body(carry, _):
            params, opt_state, key = carry
            key, sample_key = jax.random.split(key)
            batch = _sample_batch(buffer, sample_key, BATCH)
            params, opt_state, info = _ddqn_update(
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


def run_anakin_dqn(
    total_timesteps: int, seed: int, log_dir: str, num_envs: int = 64
) -> dict:
    """Train a DQN end-to-end in JAX on the wrapped PuckWorld env.

    Anakin-style: vmapped envs, a jittable rollout of ``ROLLOUT`` steps per
    iteration (collected via ``lax.scan``), and a jitted DDQN update sampling
    minibatches of ``BATCH`` from an in-memory replay buffer. Episode returns
    are read from ``timestep.extras["episode_metrics"]`` (populated by
    ``RecordEpisodeMetrics``), masked by its ``is_terminal_step`` flag.

    Returns the same metric schema as ``train_sb3.run_sb3_dqn``.
    """
    env = build_wrapped_env(num_envs)
    optimizer = optax.adam(LR)

    key = jax.random.PRNGKey(seed)
    key, params_key, reset_key = jax.random.split(key, 3)
    params = init_mlp_params(params_key)
    target_params = jax.tree_util.tree_map(lambda x: x.copy(), params)
    opt_state = optimizer.init(params)
    buffer = init_replay_buffer(BUFFER_SIZE)

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

    returns: list[float] = []
    steps: list[int] = []
    wall_clock: list[float] = []

    start = time.perf_counter()
    env_step = jnp.asarray(0, dtype=jnp.int32)
    env_step_host = 0

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

        # Extract completed-episode returns from this rollout chunk, in order.
        ep_returns_np = np.asarray(ep_returns)
        ep_dones_np = np.asarray(ep_dones)
        now = time.perf_counter()
        chunk_start_step = env_step_host - num_envs * ROLLOUT
        step_grid = chunk_start_step + num_envs * np.arange(1, ROLLOUT + 1)
        for t in range(ROLLOUT):
            done_mask = ep_dones_np[t]
            if not done_mask.any():
                continue
            for r in ep_returns_np[t][done_mask]:
                returns.append(float(r))
                steps.append(int(step_grid[t]))
                wall_clock.append(now - start)

        buffer_size_host = int(np.asarray(buffer.size))
        if buffer_size_host >= BATCH:
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

    elapsed = time.perf_counter() - start

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(log_dir) / "anakin_dqn_progress.csv"
    with csv_path.open("w") as fh:
        fh.write("step,wall_clock,return\n")
        for s, w, r in zip(steps, wall_clock, returns, strict=True):
            fh.write(f"{s},{w:.4f},{r:.4f}\n")

    return {
        "backend": "anakin_dqn",
        "returns": returns,
        "steps": steps,
        "wall_clock": wall_clock,
        "steps_per_sec": env_step_host / max(elapsed, 1e-9),
        "total_wall_clock": elapsed,
    }


if __name__ == "__main__":
    metrics = run_anakin_dqn(total_timesteps=200_000, seed=0, log_dir="results")
    print(
        f"Anakin DQN: {metrics['steps_per_sec']:.0f} steps/s, "
        f"{metrics['total_wall_clock']:.1f}s, "
        f"{len(metrics['returns'])} episodes"
    )
