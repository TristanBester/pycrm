import pytest

jnp = pytest.importorskip("jax.numpy")
jax = pytest.importorskip("jax")

from pycrm.jax import JaxCrossProduct, compile_crm  # noqa: E402
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


def _make_test_env() -> JaxCrossProduct:
    crm = CRM(env_prop_enum=Events)
    return JaxCrossProduct(
        compiled_crm=compile_crm(crm),
        reset_fn=_reset,
        step_fn=_step,
        label_fn=_label,
        max_steps=5,
        obs_fn=_obs,
    )


def _reset(key):
    del key
    return jnp.asarray([0], dtype=jnp.int32)


def _step(obs, action, key):
    del obs, action, key
    return jnp.asarray([1], dtype=jnp.int32)


def _label(obs, action, next_obs):
    del action, next_obs
    return jnp.asarray([obs[0] == 0, obs[0] == 1], dtype=jnp.bool_)


def _obs(ground_obs, u, c):
    return jnp.asarray([ground_obs[0], u, c[0]], dtype=jnp.int32)
