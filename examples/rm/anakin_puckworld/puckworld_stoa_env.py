# ruff: noqa: D101, D102, D107
"""PuckWorld authored as a Stoa environment for Anakin-style training."""

from __future__ import annotations

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from stoa.env_types import StepType, TimeStep
from stoa.environment import Environment
from stoa.spaces import BoundedArraySpace, DiscreteSpace, Space

from examples.rm.anakin_puckworld import puckworld_dynamics as dyn
from examples.rm.discrete.core.machine import PuckWorldRewardMachine
from pycrm.jax import JaxCrossProduct, JaxLabellingFunction


class JaxPuckWorldState(NamedTuple):
    """Ground state for the Stoa PuckWorld env (carries its own PRNG key)."""

    ground: dyn.PuckWorldGroundState
    key: jax.Array


class JaxPuckWorld(Environment):
    """PuckWorld ground dynamics exposed as a stoa.Environment."""

    def __init__(self) -> None:
        pass

    def reset(self, rng_key, env_params=None):
        reset_key, carry_key = jax.random.split(rng_key)
        ground = dyn.reset_ground(reset_key)
        obs = dyn.ground_obs(ground)
        ts = TimeStep(StepType.FIRST, jnp.asarray(0.0), jnp.asarray(1.0), obs, {})
        return JaxPuckWorldState(ground=ground, key=carry_key), ts

    def step(self, state, action, env_params=None):
        step_key, carry_key = jax.random.split(state.key)
        ground = dyn.step_ground(state.ground, action, step_key)
        obs = dyn.ground_obs(ground)
        ts = TimeStep(StepType.MID, jnp.asarray(0.0), jnp.asarray(1.0), obs, {})
        return JaxPuckWorldState(ground=ground, key=carry_key), ts

    def observation_space(self, env_params: Optional[object] = None) -> Space:
        return BoundedArraySpace(
            (dyn.GROUND_OBS_SIZE,), jnp.float32, -jnp.inf, jnp.inf, "obs"
        )

    def action_space(self, env_params: Optional[object] = None) -> Space:
        return DiscreteSpace(dyn.NUM_ACTIONS)

    def state_space(self, env_params: Optional[object] = None) -> Space:
        raise NotImplementedError


class PuckWorldJaxLabels(JaxLabellingFunction):
    """Pure labelling function: [T_1, T_2, T_3, A] from the post-step obs."""

    def __call__(self, ground_obs, action, next_ground_obs):
        del ground_obs, action
        return dyn.labels_vector(next_ground_obs)


def _puckworld_reward_fn(
    ground, action, next_ground, u, u_next, c, c_next,
    prop_mask, counter_mask, table_reward, params,
):
    del (ground, action, u_next, c, c_next, prop_mask, counter_mask,
         table_reward, params)
    return dyn.rm_reward(u, next_ground)


def make_puckworld_cross_product(max_steps: int = dyn.MAX_STEPS) -> JaxCrossProduct:
    """Build the PuckWorld Stoa cross-product environment."""
    return JaxCrossProduct(
        ground_env=JaxPuckWorld(),
        machine=PuckWorldRewardMachine(),
        lf=PuckWorldJaxLabels(),
        max_steps=max_steps,
        reward_fn=_puckworld_reward_fn,
        allow_dynamic_rewards=True,
    )
