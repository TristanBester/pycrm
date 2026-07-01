from typing import Any

import gymnasium as gym
import numpy as np

import jax
import jax.numpy as jnp
from pycrm.jax.crossproduct import JaxCrossProductCore, JaxCrossProductState


class GymnasiumCrossProductEnv(gym.Env):
    """Gymnasium adapter around a JAX-native cross-product core."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        core: JaxCrossProductCore,
        observation_space: gym.Space,
        action_space: gym.Space,
        seed: int = 0,
        params: Any | None = None,
        to_ground_obs_fn: Any | None = None,
        info_fn: Any | None = None,
        jit: bool = True,
    ) -> None:
        """Initialise the Gymnasium adapter."""
        super().__init__()
        self.core = core
        self.observation_space = observation_space
        self.action_space = action_space
        self.params = params
        self.to_ground_obs_fn = to_ground_obs_fn or self._default_to_ground_obs
        self.info_fn = info_fn or self._default_info
        self._key = jax.random.PRNGKey(seed)
        self._state: JaxCrossProductState | None = None
        self._reset = jax.jit(core.reset) if jit else core.reset
        self._step = jax.jit(core.step) if jit else core.step

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset the adapter and return a NumPy Gymnasium observation."""
        del options
        super().reset(seed=seed)
        if seed is not None:
            self._key = jax.random.PRNGKey(seed)
        else:
            self._key, _ = jax.random.split(self._key)
        self._key, reset_key = jax.random.split(self._key)
        state, timestep = self._reset(reset_key, self.params)
        self._state = state
        return np.asarray(timestep.observation, dtype=np.float32), {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Step the adapter and return Gymnasium NumPy/scalar values."""
        if self._state is None:
            self.reset()
        assert self._state is not None
        self._key, step_key = jax.random.split(self._key)
        next_state, timestep = self._step(
            self._state,
            jnp.asarray(action, dtype=jnp.int32),
            step_key,
            self.params,
        )
        self._state = next_state
        return (
            np.asarray(timestep.observation, dtype=np.float32),
            float(np.asarray(timestep.reward)),
            bool(np.asarray(timestep.terminated)),
            bool(np.asarray(timestep.truncated)),
            self.info_fn(next_state, timestep),
        )

    def to_ground_obs(self, obs: np.ndarray) -> np.ndarray:
        """Convert product observations back to ground observations."""
        return self.to_ground_obs_fn(obs)

    def generate_counterfactual_experience(
        self, ground_obs: np.ndarray, action: Any, next_ground_obs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return NumPy counterfactual rows for SB3/SBX-style agents."""
        batch = self.core.generate_counterfactual_experience(
            jnp.asarray(ground_obs),
            jnp.asarray(action, dtype=jnp.int32),
            jnp.asarray(next_ground_obs),
            self.params,
        )
        infos = np.array([{} for _ in range(int(batch.actions.shape[0]))], dtype=object)
        return (
            np.asarray(batch.obs, dtype=np.float32),
            np.asarray(batch.actions, dtype=np.int64),
            np.asarray(batch.next_obs, dtype=np.float32),
            np.asarray(batch.rewards, dtype=np.float32),
            np.asarray(batch.dones, dtype=bool),
            infos,
        )

    def _default_to_ground_obs(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Pass to_ground_obs_fn when constructing GymnasiumCrossProductEnv."
        )

    def _default_info(
        self, state: JaxCrossProductState, timestep: Any
    ) -> dict[str, Any]:
        del timestep
        return {"u": int(np.asarray(state.u))}
