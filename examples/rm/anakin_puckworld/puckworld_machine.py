# ruff: noqa: D101, D102, D107
"""JAX-native PuckWorld reward machine for the Anakin path.

A structural mirror of ``examples/rm/discrete/core/machine.py``'s
``PuckWorldRewardMachine``: identical states, transitions, and reward values.
The difference is that the shaping (``NOT T_x``) rewards are written with
``jax.numpy`` and marked ``@jax_reward`` so ``compile_crm`` registers them for
runtime ``lax.switch`` dispatch instead of requiring a runtime override. The
machine definition therefore stays the single source of truth for rewards in
the JAX runtime, exactly as it is for the NumPy runtime.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from examples.rm.discrete.core.label import Symbol
from pycrm.automaton import RewardMachine
from pycrm.jax import jax_reward


@jax_reward
def _nav_t_1_reward(
    obs: jax.Array, action: jax.Array, next_obs: jax.Array
) -> jax.Array:
    """Shaped negative distance to target one (mirrors the NumPy machine)."""
    del obs, action
    dist = jnp.linalg.norm(next_obs[0:2] - next_obs[4:6])
    return -dist - 10.0


@jax_reward
def _nav_t_2_reward(
    obs: jax.Array, action: jax.Array, next_obs: jax.Array
) -> jax.Array:
    """Shaped negative distance to target two (mirrors the NumPy machine)."""
    del obs, action
    dist = jnp.linalg.norm(next_obs[0:2] - next_obs[6:8])
    return -dist - 5.0


@jax_reward
def _nav_t_3_reward(
    obs: jax.Array, action: jax.Array, next_obs: jax.Array
) -> jax.Array:
    """Shaped negative distance to target three (mirrors the NumPy machine)."""
    del obs, action
    dist = jnp.linalg.norm(next_obs[0:2] - next_obs[8:10])
    return -dist


class JaxPuckWorldRewardMachine(RewardMachine):
    """PuckWorld reward machine with JAX-traceable shaping rewards."""

    def __init__(self) -> None:
        super().__init__(env_prop_enum=Symbol)

    @property
    def u_0(self) -> int:
        return 0

    @property
    def encoded_configuration_size(self) -> int:
        return 4

    def _get_state_transition_function(self) -> dict:
        return {
            0: {"T_1": 1, "NOT T_1": 0},
            1: {"T_2": 2, "NOT T_2": 1},
            2: {"T_3": -1, "NOT T_3": 2},
        }

    def _get_reward_transition_function(self) -> dict:
        return {
            0: {"T_1": 10, "NOT T_1": _nav_t_1_reward},
            1: {"T_2": 10, "NOT T_2": _nav_t_2_reward},
            2: {"T_3": 1000, "NOT T_3": _nav_t_3_reward},
        }
