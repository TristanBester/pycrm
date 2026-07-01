"""Native Stoa cross-product environment for Anakin-style JAX training."""

from typing import Any, Optional

from stoa.env_types import StepType, TimeStep
from stoa.environment import Environment
from stoa.spaces import BoundedArraySpace, Space

import jax
import jax.numpy as jnp
from pycrm.automaton import CountingRewardMachine, RewardMachine
from pycrm.jax.crm import compile_crm
from pycrm.jax.crossproduct import JaxCrossProductCore
from pycrm.jax.labelling import JaxLabellingFunction


class JaxCrossProduct(Environment):
    """Cross-product of a Stoa ground environment and a reward machine.

    The result is itself a ``stoa.Environment`` and is consumable by Stoix's
    Anakin path. It mirrors the pure-Python ``pycrm.crossproduct.CrossProduct``
    but emits ``stoa.TimeStep`` values and runs entirely in JAX.
    """

    def __init__(
        self,
        ground_env: Environment,
        machine: CountingRewardMachine | RewardMachine,
        lf: JaxLabellingFunction,
        max_steps: int,
        *,
        obs_fn: Any | None = None,
        reward_fn: Any | None = None,
        allow_dynamic_rewards: bool = False,
        discount: float = 1.0,
    ) -> None:
        """Initialise the Stoa cross-product environment.

        Args:
            ground_env: The ground environment, authored as a ``stoa.Environment``.
            machine: A reward machine or counting reward machine.
            lf: A pure JAX labelling function.
            max_steps: Product steps before truncation.
            obs_fn: Optional product observation function
                ``obs_fn(ground_obs, u, c) -> array``. Defaults to the core's
                ``concat[ground_obs, one_hot(u), c]``.
            reward_fn: Optional runtime reward override (see ``JaxCrossProductCore``).
            allow_dynamic_rewards: Passed to ``compile_crm`` for behaviour-shaped
                rewards supplied at runtime via ``reward_fn``.
            discount: Discount emitted on non-terminal timesteps.
        """
        self._ground_env = ground_env
        self._lf = lf
        self._compiled = compile_crm(
            machine, allow_dynamic_rewards=allow_dynamic_rewards
        )
        self._core = JaxCrossProductCore(
            compiled_crm=self._compiled,
            reset_fn=self._ground_reset,
            step_fn=self._ground_step,
            label_fn=lambda o, a, no, params: lf(o, a, no),
            ground_obs_fn=lambda bundle, params: bundle[1],
            obs_fn=obs_fn,
            reward_fn=reward_fn,
            max_steps=max_steps,
            discount=discount,
        )
        # The ground stoa env carries its own key; the product needs none.
        self._unused_key = jax.random.PRNGKey(0)

        # Derive the product observation shape from the default/overridden obs_fn.
        ground_shape = self._ground_env.observation_space().shape
        dummy_ground = jnp.zeros(ground_shape, dtype=jnp.float32)
        product_obs = self._core.obs_fn(
            dummy_ground,
            jnp.asarray(self._compiled.u_0, dtype=jnp.int32),
            jnp.asarray(self._compiled.c_0, dtype=jnp.int32),
        )
        self._obs_shape = tuple(int(dim) for dim in product_obs.shape)

    # -- core adapter fns ---------------------------------------------------

    def _ground_reset(self, key: Any, params: Any) -> tuple[Any, Any]:
        ground_state, timestep = self._ground_env.reset(key, params)
        return (ground_state, timestep.observation)

    def _ground_step(self, bundle: Any, action: Any, key: Any, params: Any) -> tuple:
        del key  # the stoa ground env manages its own randomness in its state
        ground_state, _ = bundle
        next_ground_state, timestep = self._ground_env.step(
            ground_state, action, params
        )
        return (next_ground_state, timestep.observation)

    # -- stoa.Environment API ----------------------------------------------

    def reset(self, rng_key: Any, env_params: Optional[Any] = None):
        """Reset and return ``(state, stoa.TimeStep)``."""
        state, jts = self._core.reset(rng_key, env_params)
        return state, self._to_stoa(jts, first=True)

    def step(self, state: Any, action: Any, env_params: Optional[Any] = None):
        """Step and return ``(state, stoa.TimeStep)``."""
        next_state, jts = self._core.step(
            state, action, self._unused_key, env_params
        )
        return next_state, self._to_stoa(jts, first=False)

    def observation_space(self, env_params: Optional[Any] = None) -> Space:
        """Return the flat product observation space."""
        return BoundedArraySpace(
            shape=self._obs_shape,
            dtype=jnp.float32,
            minimum=-jnp.inf,
            maximum=jnp.inf,
            name="observation",
        )

    def action_space(self, env_params: Optional[Any] = None) -> Space:
        """Return the ground environment's action space."""
        return self._ground_env.action_space(env_params)

    def state_space(self, env_params: Optional[Any] = None) -> Space:
        """The product state is a JAX pytree, not a flat space."""
        raise NotImplementedError(
            "JaxCrossProduct does not expose a flat state space."
        )

    # -- helpers ------------------------------------------------------------

    def _to_stoa(self, jts: Any, *, first: bool) -> TimeStep:
        if first:
            step_type = StepType.FIRST
        else:
            step_type = jnp.where(
                jts.terminated,
                StepType.TERMINATED,
                jnp.where(jts.truncated, StepType.TRUNCATED, StepType.MID),
            ).astype(jnp.int8)
        extras = {
            "u": jts.extras.u,
            "c": jts.extras.c,
            "prop_mask": jts.extras.prop_mask,
            "counter_mask": jts.extras.counter_mask,
            "transition_valid": jts.extras.transition_valid,
        }
        return TimeStep(
            step_type=step_type,
            reward=jnp.asarray(jts.reward, dtype=jnp.float32),
            discount=jnp.asarray(jts.discount, dtype=jnp.float32),
            observation=jts.observation,
            extras=extras,
        )
