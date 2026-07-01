import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest

jnp = pytest.importorskip("jax.numpy")
jax = pytest.importorskip("jax")
pytest.importorskip("stoa")

from typing import Any, NamedTuple  # noqa: E402

from stoa.env_types import StepType, TimeStep  # noqa: E402
from stoa.environment import Environment  # noqa: E402
from stoa.spaces import BoundedArraySpace, DiscreteSpace  # noqa: E402

from pycrm.jax import JaxCrossProduct, JaxLabellingFunction  # noqa: E402
from tests.crossproduct.conftest import CRM, Events  # noqa: E402


class _ToyState(NamedTuple):
    obs: Any
    key: Any


class _ToyGround(Environment):
    """Deterministic ground stoa env: obs [0] then [1] forever."""

    def __init__(self) -> None:
        pass

    def reset(
        self, rng_key: Any, env_params: Any | None = None
    ) -> tuple[_ToyState, Any]:
        del env_params
        obs = jnp.asarray([0], dtype=jnp.int32)
        ts = TimeStep(StepType.FIRST, jnp.asarray(0.0), jnp.asarray(1.0), obs, {})
        return _ToyState(obs=obs, key=rng_key), ts

    def step(
        self, state: Any, action: Any, env_params: Any | None = None
    ) -> tuple[_ToyState, Any]:
        del action, env_params
        obs = jnp.asarray([1], dtype=jnp.int32)
        ts = TimeStep(StepType.MID, jnp.asarray(0.0), jnp.asarray(1.0), obs, {})
        return _ToyState(obs=obs, key=state.key), ts

    def observation_space(self, env_params=None):
        return BoundedArraySpace((1,), jnp.int32, -10, 10, "obs")

    def action_space(self, env_params=None):
        return DiscreteSpace(1)

    def state_space(self, env_params=None):
        raise NotImplementedError


class _ToyLabels(JaxLabellingFunction):
    def __call__(self, ground_obs, action, next_ground_obs):
        del action, next_ground_obs
        return jnp.asarray([ground_obs[0] == 0, ground_obs[0] == 1], dtype=jnp.bool_)


def _obs(ground_obs, u, c):
    return jnp.asarray([ground_obs[0], u, c[0]], dtype=jnp.int32)


def _make_env() -> JaxCrossProduct:
    return JaxCrossProduct(
        ground_env=_ToyGround(),
        machine=CRM(env_prop_enum=Events),
        lf=_ToyLabels(),
        max_steps=5,
        obs_fn=_obs,
    )


def test_reset_is_first_timestep() -> None:
    """Reset returns a FIRST stoa timestep with the product observation."""
    env = _make_env()
    state, ts = jax.jit(env.reset)(jax.random.PRNGKey(0))
    assert int(ts.step_type) == int(StepType.FIRST)
    assert ts.observation.tolist() == [0, 0, 0]
    assert float(ts.reward) == 0.0
    assert isinstance(ts.extras, dict)
    assert int(ts.extras["u"]) == 0


def test_step_is_stoa_timestep() -> None:
    """Step returns a MID stoa timestep with reward, discount, and extras."""
    env = _make_env()

    @jax.jit
    def run(key):
        state, _ = env.reset(key)
        return env.step(state, jnp.asarray(0, dtype=jnp.int32))

    state, ts = run(jax.random.PRNGKey(0))
    assert int(ts.step_type) == int(StepType.MID)
    assert ts.observation.tolist() == [1, 1, 1]
    assert float(ts.reward) == 1.0
    assert float(ts.discount) == 1.0
    assert int(ts.extras["u"]) == 1
    assert bool(ts.extras["transition_valid"])


def test_observation_and_action_spaces() -> None:
    """The product exposes the flat observation and ground action spaces."""
    env = _make_env()
    assert env.observation_space().shape == (3,)
    assert isinstance(env.action_space(), DiscreteSpace)


def test_vmap_over_envs() -> None:
    """Reset and step vmap cleanly over a batch of PRNG keys."""
    env = _make_env()

    @jax.jit
    def run(keys):
        states, _ = jax.vmap(env.reset)(keys)
        return jax.vmap(lambda s: env.step(s, jnp.asarray(0, dtype=jnp.int32)))(states)

    states, ts = run(jax.random.split(jax.random.PRNGKey(0), 4))
    assert ts.observation.shape == (4, 3)
    assert ts.reward.tolist() == [1.0, 1.0, 1.0, 1.0]


def test_survives_stoix_wrapper_chain_under_scan() -> None:
    """The env runs inside AddRNGKey -> RecordEpisodeMetrics -> AutoReset -> Vmap."""
    from stoa import AutoResetWrapper, RecordEpisodeMetrics
    from stoa.core_wrappers.vmap import VmapWrapper
    from stoa.core_wrappers.wrapper import AddRNGKey

    env = _make_env()
    env = AddRNGKey(env)
    env = RecordEpisodeMetrics(env)
    env = AutoResetWrapper(env, next_obs_in_extras=True)
    env = VmapWrapper(env)

    num_envs, rollout = 3, 6
    keys = jax.random.split(jax.random.PRNGKey(0), num_envs)

    @jax.jit
    def rollout_fn(keys):
        state, ts = env.reset(keys)

        def body(carry, _):
            state = carry
            actions = jnp.zeros((num_envs,), dtype=jnp.int32)
            state, ts = env.step(state, actions)
            return state, ts.reward

        return jax.lax.scan(body, state, None, length=rollout)

    _, rewards = rollout_fn(keys)
    assert rewards.shape == (rollout, num_envs)
