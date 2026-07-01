# ruff: noqa: D101, D102, D107
"""LetterWorld authored as a Stoa environment for Anakin-style training."""

from __future__ import annotations

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from stoa.env_types import StepType, TimeStep
from stoa.environment import Environment
from stoa.spaces import BoundedArraySpace, DiscreteSpace, Space

from examples.introduction.core.machine import LetterWorldCountingRewardMachine
from examples.rm.letterworld_anakin import letterworld_dynamics as dyn
from pycrm.jax import JaxCrossProduct, JaxLabellingFunction


class JaxLetterWorldState(NamedTuple):
    ground: dyn.LetterWorldGroundState
    key: jax.Array


class JaxLetterWorld(Environment):
    """LetterWorld ground dynamics exposed as a stoa.Environment (owns its PRNG key)."""

    def __init__(self) -> None:
        pass

    def reset(self, rng_key, env_params=None):
        reset_key, carry_key = jax.random.split(rng_key)
        ground = dyn.reset_ground(reset_key)
        obs = dyn.ground_obs(ground)
        ts = TimeStep(StepType.FIRST, jnp.asarray(0.0), jnp.asarray(1.0), obs, {})
        return JaxLetterWorldState(ground=ground, key=carry_key), ts

    def step(self, state, action, env_params=None):
        step_key, carry_key = jax.random.split(state.key)
        ground = dyn.step_ground(state.ground, action, step_key)
        obs = dyn.ground_obs(ground)
        ts = TimeStep(StepType.MID, jnp.asarray(0.0), jnp.asarray(1.0), obs, {})
        return JaxLetterWorldState(ground=ground, key=carry_key), ts

    def observation_space(self, env_params: Optional[object] = None) -> Space:
        return BoundedArraySpace(
            (dyn.GROUND_OBS_SIZE,), jnp.int32, 0, 100, "obs"
        )

    def action_space(self, env_params: Optional[object] = None) -> Space:
        return DiscreteSpace(dyn.NUM_ACTIONS)

    def state_space(self, env_params: Optional[object] = None) -> Space:
        raise NotImplementedError


class LetterWorldJaxLabels(JaxLabellingFunction):
    """Pure labelling function: [A, B, C] from the post-step ground observation."""

    def __call__(self, ground_obs, action, next_ground_obs):
        del ground_obs, action
        return dyn.labels_vector(next_ground_obs)


def make_letterworld_cross_product(max_steps: int = dyn.MAX_STEPS) -> JaxCrossProduct:
    """Build the LetterWorld Stoa cross-product (reward comes from the compiled CRM)."""
    return JaxCrossProduct(
        ground_env=JaxLetterWorld(),
        machine=LetterWorldCountingRewardMachine(),
        lf=LetterWorldJaxLabels(),
        max_steps=max_steps,
    )
