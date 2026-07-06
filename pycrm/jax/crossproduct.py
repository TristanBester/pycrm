from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np

import jax
import jax.numpy as jnp
from pycrm.jax.crm import JaxCompiledCRM


class JaxCrossProductState(NamedTuple):
    """Functional cross-product state carried through JAX transforms."""

    ground_state: Any
    ground_obs: Any
    u: Any
    c: Any
    steps: Any


class JaxCrossProductExtras(NamedTuple):
    """Auxiliary product-state fields exposed on JAX timesteps."""

    ground_obs: Any
    u: Any
    c: Any
    prop_mask: Any
    counter_mask: Any
    transition_valid: Any


class JaxTimeStep(NamedTuple):
    """JAX-native timestep with Stoa/Jumanji-style transition signals."""

    observation: Any
    reward: Any
    terminated: Any
    truncated: Any
    discount: Any
    valid: Any
    extras: JaxCrossProductExtras

    def last(self) -> Any:
        """Return whether this timestep ends an episode."""
        return jnp.logical_or(self.terminated, self.truncated)


class CounterfactualBatch(NamedTuple):
    """Fixed-shape counterfactual transition rows."""

    obs: Any
    actions: Any
    next_obs: Any
    rewards: Any
    dones: Any
    valid: Any


class JaxCrossProductCore:
    """Pure functional cross-product engine for end-to-end JAX training."""

    def __init__(
        self,
        *,
        compiled_crm: JaxCompiledCRM,
        reset_fn: Callable[[Any, Any], Any],
        step_fn: Callable[[Any, Any, Any, Any], Any],
        label_fn: Callable[[Any, Any, Any, Any], Any],
        max_steps: int,
        ground_obs_fn: Callable[[Any, Any], Any] | None = None,
        obs_fn: Callable[[Any, Any, Any], Any] | None = None,
        reward_fn: Callable[
            [Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any], Any
        ]
        | None = None,
        discount: float = 1.0,
    ) -> None:
        """Initialise the JAX-native cross-product core.

        Args:
            compiled_crm: Dense CRM table returned by ``compile_crm``.
            reset_fn: JAX function ``reset_fn(key, params) -> ground_state``.
            step_fn: JAX function
                ``step_fn(ground_state, action, key, params) -> ground_state``.
            label_fn: JAX function returning one boolean per CRM proposition.
            max_steps: Number of product steps before truncation.
            ground_obs_fn: Optional ``ground_obs_fn(ground_state, params)``.
            obs_fn: Optional product observation function.
            reward_fn: Optional JAX reward override. It receives previous ground
                observation, action, next ground observation, current/next CRM
                state and counters, proposition/counter masks, table reward, and
                params.
            discount: Discount emitted on non-terminal timesteps.
        """
        self.max_steps = int(max_steps)
        self.reset_fn = reset_fn
        self.step_fn = step_fn
        self.label_fn = label_fn
        self.ground_obs_fn = ground_obs_fn or self._identity_ground_obs
        self.obs_fn = obs_fn or self._default_obs
        self.discount = float(discount)

        self.u_0 = np.asarray(compiled_crm.u_0, dtype=np.int32)
        self.c_0 = np.asarray(compiled_crm.c_0, dtype=np.int32)
        self.terminal_mask = np.asarray(compiled_crm.terminal_mask, dtype=np.bool_)
        self.next_state = np.asarray(compiled_crm.next_state, dtype=np.int32)
        self.counter_delta = np.asarray(compiled_crm.counter_delta, dtype=np.int32)
        self.reward = np.asarray(compiled_crm.reward, dtype=np.float32)
        self.valid = np.asarray(compiled_crm.valid, dtype=np.bool_)
        self.counterfactual_machine_states = np.asarray(
            compiled_crm.counterfactual_machine_states, dtype=np.int32
        )
        self.counterfactual_counter_configurations = np.asarray(
            compiled_crm.counterfactual_counter_configurations, dtype=np.int32
        )
        self.num_props = compiled_crm.num_props
        self.num_counters = compiled_crm.num_counters
        self.num_machine_states = compiled_crm.num_machine_states
        self._prop_bit_values = np.asarray(
            [1 << idx for idx in range(self.num_props)], dtype=np.int32
        )
        self._counter_bit_values = np.asarray(
            [1 << idx for idx in range(self.num_counters)], dtype=np.int32
        )

        # Dynamic reward dispatch: @jax_reward-marked transitions carry a
        # non-zero id in ``reward_fn_id`` selecting a branch in ``reward_fns``.
        self.reward_fn_id = np.asarray(
            getattr(compiled_crm, "reward_fn_id", np.zeros_like(self.reward, np.int32)),
            dtype=np.int32,
        )
        self.reward_fns = tuple(getattr(compiled_crm, "reward_fns", ()))
        self._reward_branches = (
            self._branch_table_reward,
            *(self._make_reward_branch(fn) for fn in self.reward_fns),
        )

        # An explicit override wins. Otherwise dispatch through the registry when
        # the machine has dynamic rewards, else return the scalar table value.
        if reward_fn is not None:
            self.reward_fn = reward_fn
        elif self.reward_fns:
            self.reward_fn = self._dispatch_reward
        else:
            self.reward_fn = self._table_reward

    def reset(
        self, key: Any, params: Any | None = None
    ) -> tuple[JaxCrossProductState, JaxTimeStep]:
        """Reset the cross-product and return ``(state, timestep)``."""
        ground_state = self.reset_fn(key, params)
        ground_observation = self.ground_obs_fn(ground_state, params)
        state = JaxCrossProductState(
            ground_state=ground_state,
            ground_obs=ground_observation,
            u=self.u_0,
            c=self.c_0,
            steps=jnp.asarray(0, dtype=jnp.int32),
        )
        timestep = self._make_timestep(
            observation=self.obs_fn(state.ground_obs, state.u, state.c),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            terminated=jnp.asarray(False),
            truncated=jnp.asarray(False),
            valid=jnp.asarray(True),
            ground_obs=state.ground_obs,
            u=state.u,
            c=state.c,
            prop_mask=jnp.asarray(0, dtype=jnp.int32),
            counter_mask=self._counters_to_mask(state.c),
        )
        return state, timestep

    def step(
        self,
        state: JaxCrossProductState,
        action: Any,
        key: Any,
        params: Any | None = None,
    ) -> tuple[JaxCrossProductState, JaxTimeStep]:
        """Step the cross-product and return ``(state, timestep)``."""
        next_ground_state = self.step_fn(state.ground_state, action, key, params)
        next_ground_obs = self.ground_obs_fn(next_ground_state, params)
        props = self.label_fn(state.ground_obs, action, next_ground_obs, params)
        prop_mask = self._props_to_mask(props)
        counter_mask = self._counters_to_mask(state.c)

        valid = jnp.asarray(self.valid)
        next_state_table = jnp.asarray(self.next_state)
        counter_delta_table = jnp.asarray(self.counter_delta)
        reward_table = jnp.asarray(self.reward)
        terminal_mask = jnp.asarray(self.terminal_mask)

        transition_valid = valid[state.u, prop_mask, counter_mask]
        u_next = next_state_table[state.u, prop_mask, counter_mask]
        c_delta = counter_delta_table[state.u, prop_mask, counter_mask]
        table_reward = reward_table[state.u, prop_mask, counter_mask]

        u_next = jnp.where(transition_valid, u_next, state.u)
        c_next = jnp.where(transition_valid, state.c + c_delta, state.c)
        reward = self.reward_fn(
            state.ground_obs,
            action,
            next_ground_obs,
            state.u,
            u_next,
            state.c,
            c_next,
            prop_mask,
            counter_mask,
            table_reward,
            params,
        )
        reward = jnp.where(
            transition_valid,
            jnp.asarray(reward, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        steps_next = state.steps + jnp.asarray(1, dtype=jnp.int32)
        next_state = JaxCrossProductState(
            ground_state=next_ground_state,
            ground_obs=next_ground_obs,
            u=u_next,
            c=c_next,
            steps=steps_next,
        )
        terminated = terminal_mask[u_next]
        truncated = steps_next >= self.max_steps
        timestep = self._make_timestep(
            observation=self.obs_fn(next_state.ground_obs, next_state.u, next_state.c),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            valid=transition_valid,
            ground_obs=next_state.ground_obs,
            u=next_state.u,
            c=next_state.c,
            prop_mask=prop_mask,
            counter_mask=counter_mask,
        )
        return next_state, timestep

    def generate_counterfactual_experience(
        self,
        ground_obs: Any,
        action: Any,
        next_ground_obs: Any,
        params: Any | None = None,
    ) -> CounterfactualBatch:
        """Generate fixed-shape JAX arrays of counterfactual experiences."""
        props = self.label_fn(ground_obs, action, next_ground_obs, params)
        prop_mask = self._props_to_mask(props)

        counterfactual_machine_states = jnp.asarray(self.counterfactual_machine_states)
        counterfactual_counter_configurations = jnp.asarray(
            self.counterfactual_counter_configurations
        )
        valid_table = jnp.asarray(self.valid)
        next_state_table = jnp.asarray(self.next_state)
        counter_delta_table = jnp.asarray(self.counter_delta)
        reward_table = jnp.asarray(self.reward)
        terminal_mask = jnp.asarray(self.terminal_mask)

        u_grid = jnp.repeat(
            counterfactual_machine_states,
            counterfactual_counter_configurations.shape[0],
            axis=0,
        )
        c_grid = jnp.tile(
            counterfactual_counter_configurations,
            (counterfactual_machine_states.shape[0], 1),
        )
        counter_mask = jax.vmap(self._counters_to_mask)(c_grid)

        valid = valid_table[u_grid, prop_mask, counter_mask]
        u_next = next_state_table[u_grid, prop_mask, counter_mask]
        c_delta = counter_delta_table[u_grid, prop_mask, counter_mask]
        c_next = c_grid + c_delta
        table_reward = reward_table[u_grid, prop_mask, counter_mask]

        def row_reward(
            u: Any,
            next_u: Any,
            c: Any,
            next_c: Any,
            c_mask: Any,
            transition_reward: Any,
        ) -> Any:
            return self.reward_fn(
                ground_obs,
                action,
                next_ground_obs,
                u,
                next_u,
                c,
                next_c,
                prop_mask,
                c_mask,
                transition_reward,
                params,
            )

        rewards = jax.vmap(row_reward)(
            u_grid,
            u_next,
            c_grid,
            c_next,
            counter_mask,
            table_reward,
        ).astype(jnp.float32)
        rewards = jnp.where(valid, rewards, jnp.asarray(0.0, dtype=jnp.float32))
        done = terminal_mask[u_next]

        obs = jax.vmap(lambda u, c: self.obs_fn(ground_obs, u, c))(u_grid, c_grid)
        next_obs = jax.vmap(lambda u, c: self.obs_fn(next_ground_obs, u, c))(
            u_next, c_next
        )
        actions = jnp.full((u_grid.shape[0],), action, dtype=jnp.asarray(action).dtype)
        return CounterfactualBatch(obs, actions, next_obs, rewards, done, valid)

    def _make_timestep(
        self,
        *,
        observation: Any,
        reward: Any,
        terminated: Any,
        truncated: Any,
        valid: Any,
        ground_obs: Any,
        u: Any,
        c: Any,
        prop_mask: Any,
        counter_mask: Any,
    ) -> JaxTimeStep:
        discount = jnp.where(
            jnp.logical_or(terminated, truncated),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(self.discount, dtype=jnp.float32),
        )
        return JaxTimeStep(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            discount=discount,
            valid=valid,
            extras=JaxCrossProductExtras(
                ground_obs=ground_obs,
                u=u,
                c=c,
                prop_mask=prop_mask,
                counter_mask=counter_mask,
                transition_valid=valid,
            ),
        )

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

    def _identity_ground_obs(self, ground_state: Any, params: Any | None) -> Any:
        del params
        return ground_state

    def _default_obs(self, ground_obs: Any, u: Any, c: Any) -> Any:
        ground_obs = jnp.asarray(ground_obs, dtype=jnp.float32)
        u_enc = jax.nn.one_hot(u, self.num_machine_states, dtype=jnp.float32)
        c_enc = jnp.asarray(c, dtype=jnp.float32)
        return jnp.concatenate((ground_obs, u_enc, c_enc), axis=0)

    def _table_reward(
        self,
        ground_obs: Any,
        action: Any,
        next_ground_obs: Any,
        u: Any,
        u_next: Any,
        c: Any,
        c_next: Any,
        prop_mask: Any,
        counter_mask: Any,
        table_reward: Any,
        params: Any | None,
    ) -> Any:
        del (
            ground_obs,
            action,
            next_ground_obs,
            u,
            u_next,
            c,
            c_next,
            prop_mask,
            counter_mask,
            params,
        )
        return table_reward

    def _dispatch_reward(
        self,
        ground_obs: Any,
        action: Any,
        next_ground_obs: Any,
        u: Any,
        u_next: Any,
        c: Any,
        c_next: Any,
        prop_mask: Any,
        counter_mask: Any,
        table_reward: Any,
        params: Any | None,
    ) -> Any:
        """Route to the reward branch selected by ``reward_fn_id``.

        Branch ``0`` returns the scalar table reward; branch ``k`` runs the
        ``k``-th registered ``@jax_reward`` callable. Under ``vmap`` (used by
        counterfactual generation) the branch index is batched, so JAX evaluates
        every branch per row and selects — cheap while the registry is small.
        """
        reward_id = jnp.asarray(self.reward_fn_id)[u, prop_mask, counter_mask]
        return jax.lax.switch(
            reward_id,
            self._reward_branches,
            ground_obs,
            action,
            next_ground_obs,
            u,
            u_next,
            c,
            c_next,
            prop_mask,
            counter_mask,
            table_reward,
            params,
        )

    def _branch_table_reward(
        self,
        ground_obs: Any,
        action: Any,
        next_ground_obs: Any,
        u: Any,
        u_next: Any,
        c: Any,
        c_next: Any,
        prop_mask: Any,
        counter_mask: Any,
        table_reward: Any,
        params: Any | None,
    ) -> Any:
        """Dispatch branch ``0``: emit the precompiled scalar table reward."""
        del (
            ground_obs,
            action,
            next_ground_obs,
            u,
            u_next,
            c,
            c_next,
            prop_mask,
            counter_mask,
            params,
        )
        return jnp.asarray(table_reward, dtype=jnp.float32)

    def _make_reward_branch(self, reward_fn: Callable) -> Callable:
        """Wrap a CRM reward callable as a uniform ``lax.switch`` branch.

        The registered callable follows the CRM reward contract
        ``(obs, action, next_obs) -> reward``; the extra machine-state operands
        are accepted and ignored so every branch shares one signature.
        """

        def branch(
            ground_obs: Any,
            action: Any,
            next_ground_obs: Any,
            u: Any,
            u_next: Any,
            c: Any,
            c_next: Any,
            prop_mask: Any,
            counter_mask: Any,
            table_reward: Any,
            params: Any | None,
        ) -> Any:
            del u, u_next, c, c_next, prop_mask, counter_mask, table_reward, params
            return jnp.asarray(
                reward_fn(ground_obs, action, next_ground_obs), dtype=jnp.float32
            )

        return branch


class FunctionalJaxCrossProduct:
    """Compatibility facade exposing the original tuple-based JAX API."""

    def __init__(
        self,
        compiled_crm: JaxCompiledCRM,
        reset_fn: Callable[[Any], Any],
        step_fn: Callable[[Any, Any, Any], Any],
        label_fn: Callable[[Any, Any, Any], Any],
        max_steps: int,
        obs_fn: Callable[[Any, Any, Any], Any] | None = None,
    ) -> None:
        """Initialise the compatibility wrapper around ``JaxCrossProductCore``."""

        def core_reset(key: Any, params: Any | None) -> Any:
            del params
            return reset_fn(key)

        def core_step(
            ground_state: Any,
            action: Any,
            key: Any,
            params: Any | None,
        ) -> Any:
            del params
            return step_fn(ground_state, action, key)

        def core_label(
            ground_obs: Any,
            action: Any,
            next_ground_obs: Any,
            params: Any | None,
        ) -> Any:
            del params
            return label_fn(ground_obs, action, next_ground_obs)

        self.core = JaxCrossProductCore(
            compiled_crm=compiled_crm,
            reset_fn=core_reset,
            step_fn=core_step,
            label_fn=core_label,
            max_steps=max_steps,
            obs_fn=obs_fn,
        )

    def reset(self, key: Any) -> tuple[JaxCrossProductState, Any]:
        """Reset and return the original ``(state, observation)`` shape."""
        state, timestep = self.core.reset(key)
        return state, timestep.observation

    def step(
        self, state: JaxCrossProductState, action: Any, key: Any
    ) -> tuple[JaxCrossProductState, Any, Any, Any, Any, Any]:
        """Step and return the original tuple-based transition shape."""
        next_state, timestep = self.core.step(state, action, key)
        return (
            next_state,
            timestep.observation,
            timestep.reward,
            timestep.terminated,
            timestep.truncated,
            timestep.valid,
        )

    def generate_counterfactual_experience(
        self, ground_obs: Any, action: Any, next_ground_obs: Any
    ) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Generate counterfactual rows using the original tuple shape."""
        batch = self.core.generate_counterfactual_experience(
            ground_obs, action, next_ground_obs
        )
        return (
            batch.obs,
            batch.actions,
            batch.next_obs,
            batch.rewards,
            batch.dones,
            batch.valid,
        )
