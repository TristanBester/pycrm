# ruff: noqa: D101
"""Pure JAX PuckWorld dynamics for the anakin_puckworld demo.

This module re-hosts the proven pure functions from
``examples/rm/discrete-fast/jax_puckworld.py`` verbatim, giving the
acceptance-demo phase a stable, self-contained API to build the Stoa ground
env (B2) and reward/counterfactual parity tests (B3) on top of.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

GROUND_OBS_SIZE = 12
PRODUCT_OBS_SIZE = 17
NUM_ACTIONS = 4
NUM_MACHINE_STATES = 4
MAX_STEPS = 1000
TARGET_THRESHOLD = 0.15
ADVERSARY_THRESHOLD = 0.1
TARGET_SEPARATION = 0.45
TARGET_SAMPLE_CANDIDATES = 64

RIGHT = 0
LEFT = 1
UP = 2
DOWN = 3


class PuckWorldGroundState(NamedTuple):
    agent_pos: jax.Array
    agent_vel: jax.Array
    target_one_pos: jax.Array
    target_two_pos: jax.Array
    target_three_pos: jax.Array
    adversary_pos: jax.Array


def _sample_target(key: jax.Array, other_a: jax.Array, other_b: jax.Array) -> jax.Array:
    """Sample a target location matching the Python rejection rule."""
    samples = jax.random.uniform(
        key,
        (TARGET_SAMPLE_CANDIDATES, 2),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    ok = jnp.logical_and(
        jnp.linalg.norm(samples - other_a, axis=-1) > TARGET_SEPARATION,
        jnp.linalg.norm(samples - other_b, axis=-1) > TARGET_SEPARATION,
    )
    first_ok = jnp.argmax(ok.astype(jnp.int32))
    fallback = samples[-1]
    return jnp.where(jnp.any(ok), samples[first_ok], fallback).astype(jnp.float32)


def reset_ground(
    key: jax.Array,
    params: object | None = None,
) -> PuckWorldGroundState:
    """Reset one JAX PuckWorld ground environment."""
    del params
    k1, k2, k3 = jax.random.split(key, 3)
    zeros = jnp.zeros((2,), dtype=jnp.float32)
    target_one = _sample_target(k1, zeros, zeros)
    target_two = _sample_target(k2, target_one, zeros)
    target_three = _sample_target(k3, target_one, target_two)
    return PuckWorldGroundState(
        agent_pos=zeros,
        agent_vel=zeros,
        target_one_pos=target_one,
        target_two_pos=target_two,
        target_three_pos=target_three,
        adversary_pos=jnp.asarray([0.8, 0.8], dtype=jnp.float32),
    )


def _wrap_position(pos: jax.Array) -> jax.Array:
    return jnp.where(pos < -1.0, 1.0, jnp.where(pos > 1.0, -1.0, pos))


def _maybe_respawn_target(
    *,
    key: jax.Array,
    target: jax.Array,
    agent_pos: jax.Array,
    other_a: jax.Array,
    other_b: jax.Array,
) -> jax.Array:
    should_respawn = jnp.linalg.norm(agent_pos - target) < TARGET_THRESHOLD
    sampled = _sample_target(key, other_a, other_b)
    return jnp.where(should_respawn, sampled, target)


def step_ground(
    state: PuckWorldGroundState,
    action: jax.Array,
    key: jax.Array,
    params: object | None = None,
) -> PuckWorldGroundState:
    """Step one functional PuckWorld ground environment."""
    del params
    noise_key, t1_key, t2_key, t3_key = jax.random.split(key, 4)
    action = jnp.asarray(action, dtype=jnp.int32)
    delta = jnp.asarray(
        [
            jnp.where(action == RIGHT, 0.05, jnp.where(action == LEFT, -0.05, 0.0)),
            jnp.where(action == UP, 0.05, jnp.where(action == DOWN, -0.05, 0.0)),
        ],
        dtype=jnp.float32,
    )
    agent_vel = state.agent_vel + delta
    agent_vel = agent_vel + jax.random.normal(noise_key, (2,)) * 0.01
    agent_vel = jnp.clip(agent_vel, -0.15, 0.15).astype(jnp.float32)
    agent_pos = _wrap_position(state.agent_pos + agent_vel).astype(jnp.float32)

    direction = jnp.sign(agent_pos - state.adversary_pos)
    adversary_pos = (state.adversary_pos + direction * 0.0005).astype(jnp.float32)

    target_one = _maybe_respawn_target(
        key=t1_key,
        target=state.target_one_pos,
        agent_pos=agent_pos,
        other_a=state.target_two_pos,
        other_b=state.target_three_pos,
    )
    target_two = _maybe_respawn_target(
        key=t2_key,
        target=state.target_two_pos,
        agent_pos=agent_pos,
        other_a=target_one,
        other_b=state.target_three_pos,
    )
    target_three = _maybe_respawn_target(
        key=t3_key,
        target=state.target_three_pos,
        agent_pos=agent_pos,
        other_a=target_one,
        other_b=target_two,
    )

    return PuckWorldGroundState(
        agent_pos=agent_pos,
        agent_vel=agent_vel,
        target_one_pos=target_one,
        target_two_pos=target_two,
        target_three_pos=target_three,
        adversary_pos=adversary_pos,
    )


def ground_obs(state: PuckWorldGroundState) -> jax.Array:
    """Return the 12-dim PuckWorld ground observation from a ground state."""
    return jnp.concatenate(
        (
            state.agent_pos,
            state.agent_vel,
            state.target_one_pos,
            state.target_two_pos,
            state.target_three_pos,
            state.adversary_pos,
        ),
        axis=0,
    ).astype(jnp.float32)


def labels_vector(next_ground_obs: jax.Array) -> jax.Array:
    """Return the [T_1, T_2, T_3, A] boolean proposition vector."""
    agent_pos = next_ground_obs[0:2]
    t1 = jnp.linalg.norm(agent_pos - next_ground_obs[4:6]) < TARGET_THRESHOLD
    t2 = jnp.linalg.norm(agent_pos - next_ground_obs[6:8]) < TARGET_THRESHOLD
    t3 = jnp.linalg.norm(agent_pos - next_ground_obs[8:10]) < TARGET_THRESHOLD
    adv = jnp.linalg.norm(agent_pos - next_ground_obs[10:12]) < ADVERSARY_THRESHOLD
    return jnp.asarray([t1, t2, t3, adv], dtype=jnp.bool_)


def rm_reward(u: jax.Array, next_ground_obs: jax.Array) -> jax.Array:
    """Return the PuckWorld reward-machine reward for machine state ``u``.

    Mirrors ``examples/rm/discrete/core/machine.py::PuckWorldRewardMachine``:
    +10 on reaching T_1/T_2, +1000 on T_3, else negative shaped distance.
    """
    agent_pos = next_ground_obs[0:2]
    dist_one = jnp.linalg.norm(agent_pos - next_ground_obs[4:6])
    dist_two = jnp.linalg.norm(agent_pos - next_ground_obs[6:8])
    dist_three = jnp.linalg.norm(agent_pos - next_ground_obs[8:10])
    labels = labels_vector(next_ground_obs)
    r0 = jnp.where(labels[0], 10.0, -dist_one - 10.0)
    r1 = jnp.where(labels[1], 10.0, -dist_two - 5.0)
    r2 = jnp.where(labels[2], 1000.0, -dist_three)
    return jnp.select([u == 0, u == 1, u == 2], [r0, r1, r2], default=0.0).astype(
        jnp.float32
    )
