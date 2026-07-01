"""Pure-JAX LetterWorld dynamics (re-host of examples/introduction/core/ground.py)."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

N_ROWS = 3
N_COLS = 7
A_POSITION = jnp.array([1, 1], dtype=jnp.int32)
C_POSITION = jnp.array([1, 5], dtype=jnp.int32)
START_POSITION = jnp.array([1, 3], dtype=jnp.int32)
RIGHT, LEFT, UP, DOWN = 0, 1, 2, 3
NUM_ACTIONS = 4
GROUND_OBS_SIZE = 3
SYMBOL_FLIP_PROB = 0.5
MAX_STEPS = 100


class LetterWorldGroundState(NamedTuple):
    """Ground state: agent (row, col) and whether the symbol has been seen."""

    agent_pos: jax.Array  # int32[2] = (row, col)
    symbol_seen: jax.Array  # int32 scalar in {0, 1}


def reset_ground(key: jax.Array) -> LetterWorldGroundState:
    """Return the deterministic start state (key unused; reset is deterministic)."""
    del key
    return LetterWorldGroundState(
        agent_pos=START_POSITION,
        symbol_seen=jnp.asarray(0, dtype=jnp.int32),
    )


def _move(pos: jax.Array, action: jax.Array) -> jax.Array:
    """Move on the grid with wall-clamping, matching numpy LetterWorld."""
    row, col = pos[0], pos[1]
    right = jnp.array([row, jnp.minimum(col + 1, N_COLS - 1)], dtype=jnp.int32)
    left = jnp.array([row, jnp.maximum(col - 1, 0)], dtype=jnp.int32)
    up = jnp.array([jnp.maximum(row - 1, 0), col], dtype=jnp.int32)
    down = jnp.array([jnp.minimum(row + 1, N_ROWS - 1), col], dtype=jnp.int32)
    return jnp.select(
        [action == RIGHT, action == LEFT, action == UP, action == DOWN],
        [right, left, up, down],
        default=pos,
    ).astype(jnp.int32)


def step_ground(
    state: LetterWorldGroundState, action: jax.Array, key: jax.Array
) -> LetterWorldGroundState:
    """Move, then flip symbol_seen with p=0.5 iff arriving at A while unseen."""
    new_pos = _move(state.agent_pos, action)
    at_a = jnp.logical_and(jnp.all(new_pos == A_POSITION), state.symbol_seen == 0)
    flip = jax.random.uniform(key) < SYMBOL_FLIP_PROB
    new_seen = jnp.where(
        jnp.logical_and(at_a, flip),
        jnp.asarray(1, dtype=jnp.int32),
        state.symbol_seen,
    )
    return LetterWorldGroundState(agent_pos=new_pos, symbol_seen=new_seen)


def ground_obs(state: LetterWorldGroundState) -> jax.Array:
    """Return the 3-dim observation [symbol_seen, row, col]."""
    return jnp.concatenate(
        [jnp.reshape(state.symbol_seen, (1,)), state.agent_pos]
    ).astype(jnp.int32)


def labels_vector(next_ground_obs: jax.Array) -> jax.Array:
    """Return the [A, B, C] boolean proposition vector (order matches env_prop_enum)."""
    seen = next_ground_obs[0]
    pos = next_ground_obs[1:3]
    a = jnp.logical_and(seen == 0, jnp.all(pos == A_POSITION))
    b = jnp.logical_and(seen == 1, jnp.all(pos == A_POSITION))
    c = jnp.logical_and(seen == 1, jnp.all(pos == C_POSITION))
    return jnp.asarray([a, b, c], dtype=jnp.bool_)
