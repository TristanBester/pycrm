from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from pycrm.jax.crm import JaxCompiledCRM


class JaxCrossProductState(NamedTuple):
    """Functional cross-product state carried through JAX transforms."""

    ground_obs: Any
    u: Any
    c: Any
    steps: Any


class JaxCrossProduct:
    """Pure functional cross-product environment for JAX transforms."""

    def __init__(
        self,
        compiled_crm: JaxCompiledCRM,
        reset_fn: Callable[[Any], Any],
        step_fn: Callable[[Any, Any, Any], Any],
        label_fn: Callable[[Any, Any, Any], Any],
        max_steps: int,
        obs_fn: Callable[[Any, Any, Any], Any] | None = None,
    ) -> None:
        """Initialise the functional cross-product environment.

        Args:
            compiled_crm: Dense CRM table returned by ``compile_crm``.
            reset_fn: JAX-compatible function ``reset_fn(key) -> ground_obs``.
            step_fn: JAX-compatible function
                ``step_fn(ground_obs, action, key) -> next_ground_obs``.
            label_fn: JAX-compatible function returning a boolean vector with
                one entry per CRM proposition.
            max_steps: Number of cross-product steps before truncation.
            obs_fn: Optional function combining ``ground_obs``, ``u`` and ``c``.
        """
        super().__init__()
        self.max_steps = int(max_steps)
        self.reset_fn = reset_fn
        self.step_fn = step_fn
        self.label_fn = label_fn
        self.obs_fn = obs_fn or self._default_obs

        self.u_0 = jnp.asarray(compiled_crm.u_0, dtype=jnp.int32)
        self.c_0 = jnp.asarray(compiled_crm.c_0, dtype=jnp.int32)
        self.terminal_mask = jnp.asarray(compiled_crm.terminal_mask, dtype=jnp.bool_)
        self.next_state = jnp.asarray(compiled_crm.next_state, dtype=jnp.int32)
        self.counter_delta = jnp.asarray(compiled_crm.counter_delta, dtype=jnp.int32)
        self.reward = jnp.asarray(compiled_crm.reward, dtype=jnp.float32)
        self.valid = jnp.asarray(compiled_crm.valid, dtype=jnp.bool_)
        self.counterfactual_machine_states = jnp.asarray(
            compiled_crm.counterfactual_machine_states, dtype=jnp.int32
        )
        self.counterfactual_counter_configurations = jnp.asarray(
            compiled_crm.counterfactual_counter_configurations, dtype=jnp.int32
        )
        self.num_props = compiled_crm.num_props
        self.num_counters = compiled_crm.num_counters
        self.num_machine_states = compiled_crm.num_machine_states
        self._prop_bit_values = jnp.asarray(
            [1 << idx for idx in range(self.num_props)], dtype=jnp.int32
        )
        self._counter_bit_values = jnp.asarray(
            [1 << idx for idx in range(self.num_counters)], dtype=jnp.int32
        )

    def reset(self, key: Any) -> tuple[JaxCrossProductState, Any]:
        """Reset the functional cross-product environment."""
        ground_obs = self.reset_fn(key)
        state = JaxCrossProductState(
            ground_obs=ground_obs,
            u=self.u_0,
            c=self.c_0,
            steps=jnp.asarray(0, dtype=jnp.int32),
        )
        return state, self.obs_fn(state.ground_obs, state.u, state.c)

    def step(
        self, state: JaxCrossProductState, action: Any, key: Any
    ) -> tuple[JaxCrossProductState, Any, Any, Any, Any, Any]:
        """Step the functional cross-product environment."""
        next_ground_obs = self.step_fn(state.ground_obs, action, key)
        props = self.label_fn(state.ground_obs, action, next_ground_obs)
        prop_mask = self._props_to_mask(props)
        counter_mask = self._counters_to_mask(state.c)

        transition_valid = self.valid[state.u, prop_mask, counter_mask]
        u_next = self.next_state[state.u, prop_mask, counter_mask]
        c_delta = self.counter_delta[state.u, prop_mask, counter_mask]
        reward = self.reward[state.u, prop_mask, counter_mask]

        u_next = jnp.where(transition_valid, u_next, state.u)
        c_next = jnp.where(transition_valid, state.c + c_delta, state.c)
        reward = jnp.where(
            transition_valid, reward, jnp.asarray(0.0, dtype=jnp.float32)
        )
        steps_next = state.steps + jnp.asarray(1, dtype=jnp.int32)

        next_state = JaxCrossProductState(
            ground_obs=next_ground_obs,
            u=u_next,
            c=c_next,
            steps=steps_next,
        )
        obs = self.obs_fn(next_state.ground_obs, next_state.u, next_state.c)
        terminated = self.terminal_mask[u_next]
        truncated = steps_next >= self.max_steps
        return next_state, obs, reward, terminated, truncated, transition_valid

    def generate_counterfactual_experience(
        self, ground_obs: Any, action: Any, next_ground_obs: Any
    ) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Generate JAX arrays of counterfactual CRM experiences."""
        props = self.label_fn(ground_obs, action, next_ground_obs)
        prop_mask = self._props_to_mask(props)

        u_grid = jnp.repeat(
            self.counterfactual_machine_states,
            self.counterfactual_counter_configurations.shape[0],
            axis=0,
        )
        c_grid = jnp.tile(
            self.counterfactual_counter_configurations,
            (self.counterfactual_machine_states.shape[0], 1),
        )
        counter_mask = jax.vmap(self._counters_to_mask)(c_grid)

        valid = self.valid[u_grid, prop_mask, counter_mask]
        u_next = self.next_state[u_grid, prop_mask, counter_mask]
        c_delta = self.counter_delta[u_grid, prop_mask, counter_mask]
        c_next = c_grid + c_delta
        reward = self.reward[u_grid, prop_mask, counter_mask]
        done = self.terminal_mask[u_next]

        obs = jax.vmap(lambda u, c: self.obs_fn(ground_obs, u, c))(u_grid, c_grid)
        next_obs = jax.vmap(lambda u, c: self.obs_fn(next_ground_obs, u, c))(
            u_next, c_next
        )
        actions = jnp.full((u_grid.shape[0],), action, dtype=jnp.asarray(action).dtype)
        return obs, actions, next_obs, reward, done, valid

    def _props_to_mask(self, props: Any) -> Any:
        if self.num_props == 0:
            return jnp.asarray(0, dtype=jnp.int32)

        prop_bits = jnp.asarray(props, dtype=jnp.int32)
        return jnp.sum(prop_bits * self._prop_bit_values).astype(jnp.int32)

    def _counters_to_mask(self, counters: Any) -> Any:
        if self.num_counters == 0:
            return jnp.asarray(0, dtype=jnp.int32)

        counter_bits = (jnp.asarray(counters) != 0).astype(jnp.int32)
        return jnp.sum(counter_bits * self._counter_bit_values).astype(jnp.int32)

    def _default_obs(self, ground_obs: Any, u: Any, c: Any) -> Any:
        ground_obs = jnp.asarray(ground_obs, dtype=jnp.float32)
        u_enc = jax.nn.one_hot(
            u, self.num_machine_states, dtype=jnp.float32
        )
        c_enc = jnp.asarray(c, dtype=jnp.float32)
        return jnp.concatenate((ground_obs, u_enc, c_enc), axis=0)
