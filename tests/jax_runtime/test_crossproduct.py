import pytest

jnp = pytest.importorskip("jax.numpy")
jax = pytest.importorskip("jax")

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

from pycrm.jax import (  # noqa: E402
    FunctionalJaxCrossProduct,
    GymnasiumCrossProductEnv,
    JaxCrossProductCore,
    compile_crm,
)
from tests.crossproduct.conftest import CRM, Events  # noqa: E402


def test_jax_crossproduct_single_step_jits() -> None:
    """A functional cross-product step can be JIT compiled."""
    env = _make_test_env()

    @jax.jit
    def run_step(key):
        state, _ = env.reset(key)
        return env.step(state, jnp.asarray(0, dtype=jnp.int32), key)

    _, obs, reward, terminated, truncated, valid = run_step(jax.random.PRNGKey(0))

    assert bool(valid)
    assert not bool(terminated)
    assert not bool(truncated)
    assert float(reward) == 1.0
    assert obs.tolist() == [1, 1, 1]


def test_jax_crossproduct_rollout_scan_jits() -> None:
    """A functional cross-product rollout can run through lax.scan."""
    env = _make_test_env()

    @jax.jit
    def rollout(key):
        state, _ = env.reset(key)
        keys = jax.random.split(key, 2)

        def body(carry, step_key):
            state = carry
            state, _, reward, _, _, valid = env.step(
                state, jnp.asarray(0, dtype=jnp.int32), step_key
            )
            return state, (reward, valid)

        return jax.lax.scan(body, state, keys)

    _, (rewards, valid) = rollout(jax.random.PRNGKey(0))

    assert rewards.tolist() == [1.0, 1.0]
    assert valid.tolist() == [True, True]


def test_jax_counterfactual_experience_jits() -> None:
    """Counterfactual experience generation can be JIT compiled."""
    env = _make_test_env()

    @jax.jit
    def run_counterfactual(key):
        state, _ = env.reset(key)
        next_ground_obs = _step(state.ground_obs, jnp.asarray(0), key)
        return env.generate_counterfactual_experience(
            state.ground_obs, jnp.asarray(0, dtype=jnp.int32), next_ground_obs
        )

    obs, actions, next_obs, rewards, done, valid = run_counterfactual(
        jax.random.PRNGKey(0)
    )

    assert obs.shape == (6, 3)
    assert next_obs.shape == (6, 3)
    assert actions.tolist() == [0, 0, 0, 0, 0, 0]
    assert rewards.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert done.tolist() == [False, False, False, True, True, True]
    assert valid.tolist() == [True, True, True, True, True, True]


def test_jax_crossproduct_core_timestep_jits() -> None:
    """The JAX-native core exposes a jit-able timestep API."""
    env = _make_test_core()

    @jax.jit
    def run_step(key):
        state, reset_timestep = env.reset(key)
        state, step_timestep = env.step(state, jnp.asarray(0, dtype=jnp.int32), key)
        return reset_timestep, step_timestep

    reset_timestep, step_timestep = run_step(jax.random.PRNGKey(0))

    assert reset_timestep.observation.tolist() == [0, 0, 0]
    assert float(reset_timestep.reward) == 0.0
    assert step_timestep.observation.tolist() == [1, 1, 1]
    assert float(step_timestep.reward) == 1.0
    assert bool(step_timestep.valid)
    assert not bool(step_timestep.last())
    assert int(step_timestep.extras.u) == 1


def test_jax_crossproduct_core_vmap_steps() -> None:
    """The core reset and step API can be vectorized over environments."""
    env = _make_test_core()

    @jax.jit
    def run_batch(keys):
        states, reset_timesteps = jax.vmap(env.reset)(keys)
        actions = jnp.zeros((keys.shape[0],), dtype=jnp.int32)
        next_states, step_timesteps = jax.vmap(
            lambda state, action, key: env.step(state, action, key)
        )(states, actions, keys)
        return reset_timesteps, next_states, step_timesteps

    reset_timesteps, next_states, step_timesteps = run_batch(
        jax.random.split(jax.random.PRNGKey(0), 4)
    )

    assert reset_timesteps.observation.shape == (4, 3)
    assert step_timesteps.observation.shape == (4, 3)
    assert step_timesteps.reward.tolist() == [1.0, 1.0, 1.0, 1.0]
    assert next_states.ground_obs.tolist() == [[1], [1], [1], [1]]


def test_jax_gymnasium_adapter_smoke() -> None:
    """The Gymnasium adapter preserves SB3/SBX-style NumPy contracts."""
    env = GymnasiumCrossProductEnv(
        core=_make_test_core(),
        observation_space=gym.spaces.Box(
            low=-10,
            high=10,
            shape=(3,),
            dtype=np.float32,
        ),
        action_space=gym.spaces.Discrete(1),
        seed=0,
        to_ground_obs_fn=lambda obs: np.asarray(obs)[..., :1],
    )
    obs, _ = env.reset(seed=0)
    next_obs, reward, terminated, truncated, info = env.step(0)
    c_obs, c_actions, c_next_obs, c_rewards, c_dones, c_infos = (
        env.generate_counterfactual_experience(obs[:1], 0, next_obs[:1])
    )

    assert obs.shape == (3,)
    assert next_obs.shape == (3,)
    assert reward == 1.0
    assert not terminated
    assert not truncated
    assert info["u"] == 1
    assert c_obs.shape == (6, 3)
    assert c_next_obs.shape == (6, 3)
    assert c_actions.tolist() == [0, 0, 0, 0, 0, 0]
    assert c_rewards.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert c_dones.tolist() == [False, False, False, True, True, True]
    assert c_infos.shape == (6,)


def _make_test_env() -> FunctionalJaxCrossProduct:
    crm = CRM(env_prop_enum=Events)
    return FunctionalJaxCrossProduct(
        compiled_crm=compile_crm(crm),
        reset_fn=_reset,
        step_fn=_step,
        label_fn=_label,
        max_steps=5,
        obs_fn=_obs,
    )


def _make_test_core() -> JaxCrossProductCore:
    crm = CRM(env_prop_enum=Events)
    return JaxCrossProductCore(
        compiled_crm=compile_crm(crm),
        reset_fn=_reset_with_params,
        step_fn=_step_with_params,
        label_fn=_label_with_params,
        max_steps=5,
        obs_fn=_obs,
    )


def _reset(key):
    del key
    return jnp.asarray([0], dtype=jnp.int32)


def _reset_with_params(key, params):
    del params
    return _reset(key)


def _step(obs, action, key):
    del obs, action, key
    return jnp.asarray([1], dtype=jnp.int32)


def _step_with_params(obs, action, key, params):
    del params
    return _step(obs, action, key)


def _label(obs, action, next_obs):
    del action, next_obs
    return jnp.asarray([obs[0] == 0, obs[0] == 1], dtype=jnp.bool_)


def _label_with_params(obs, action, next_obs, params):
    del params
    return _label(obs, action, next_obs)


def _obs(ground_obs, u, c):
    return jnp.asarray([ground_obs[0], u, c[0]], dtype=jnp.int32)
