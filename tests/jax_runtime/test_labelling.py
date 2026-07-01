import pytest

jnp = pytest.importorskip("jax.numpy")
jax = pytest.importorskip("jax")

from pycrm.jax import JaxLabellingFunction  # noqa: E402


class _Labels(JaxLabellingFunction):
    def __call__(self, ground_obs, action, next_ground_obs):
        del action
        return jnp.asarray(
            [next_ground_obs[0] > ground_obs[0], next_ground_obs[0] == 0],
            dtype=jnp.bool_,
        )


def test_labelling_function_returns_bool_vector() -> None:
    """The labelling function returns a boolean proposition vector."""
    lf = _Labels()
    out = lf(jnp.asarray([0.0]), jnp.asarray(0), jnp.asarray([1.0]))
    assert out.dtype == jnp.bool_
    assert out.tolist() == [True, False]


def test_labelling_function_is_jittable() -> None:
    """The labelling function is jittable via jax.jit."""
    lf = _Labels()
    jitted = jax.jit(lf.__call__)
    out = jitted(jnp.asarray([1.0]), jnp.asarray(0), jnp.asarray([0.0]))
    assert out.tolist() == [False, True]
