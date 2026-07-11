import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
jax = pytest.importorskip("jax")
pytest.importorskip("stoa")

# Make the example package importable.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.rm.anakin_puckworld import puckworld_dynamics as dyn  # noqa: E402


def test_obs_layout_matches_numpy_puckworld() -> None:
    """The JAX ground obs uses the same 12-dim layout as the numpy env."""
    state = dyn.reset_ground(jax.random.PRNGKey(0))
    obs = dyn.ground_obs(state)
    assert obs.shape == (dyn.GROUND_OBS_SIZE,)
    # agent starts at origin, adversary at (0.8, 0.8) — same as numpy PuckWorld.
    assert obs[0:2].tolist() == [0.0, 0.0]
    assert obs[10:12].tolist() == pytest.approx([0.8, 0.8])


def test_action_semantics_match_numpy() -> None:
    """RIGHT increases vel[0]; UP increases vel[1] (deterministic part)."""
    state = dyn.reset_ground(jax.random.PRNGKey(0))
    # Disable the gaussian noise contribution by comparing means over the key.
    right = dyn.step_ground(state, jnp.asarray(0), jax.random.PRNGKey(1))
    up = dyn.step_ground(state, jnp.asarray(2), jax.random.PRNGKey(1))
    # Same noise key => the only difference is the deterministic +0.05 axis.
    assert float(right.agent_vel[0]) > float(up.agent_vel[0])
    assert float(up.agent_vel[1]) > float(right.agent_vel[1])


def test_labels_and_target_reward_match_python_machine() -> None:
    """Reaching target one fires T_1 and yields reward 10 at machine state 0."""
    from examples.rm.anakin_puckworld.puckworld_machine import (
        JaxPuckWorldRewardMachine,
    )

    obs = np.zeros(dyn.GROUND_OBS_SIZE, dtype=np.float32)
    obs[4:6] = [0.0, 0.05]  # target one within TARGET_THRESHOLD of the agent
    # target two/three and the adversary must sit away from the agent (as in a
    # real trajectory, where targets/adversary are separated from the agent),
    # otherwise they would trivially coincide with agent_pos == [0, 0].
    obs[6:8] = [0.9, 0.9]  # target two — far from the agent
    obs[8:10] = [-0.9, -0.9]  # target three — far from the agent
    obs[10:12] = [0.8, 0.8]  # adversary — matches numpy PuckWorld's reset spawn
    labels = dyn.labels_vector(jnp.asarray(obs))
    assert labels.tolist() == [True, False, False, False]
    # Reaching T_1 at machine state 0 yields the scalar target reward (+10).
    delta_r = JaxPuckWorldRewardMachine()._get_reward_transition_function()
    assert delta_r[0]["T_1"] == 10


def test_puckworld_cross_product_shapes_and_first_timestep() -> None:
    """The Stoa cross-product exposes the expected 17-dim product obs."""
    from stoa.env_types import StepType

    from examples.rm.anakin_puckworld.puckworld_stoa_env import (
        make_puckworld_cross_product,
    )

    env = make_puckworld_cross_product(max_steps=50)
    assert env.observation_space().shape == (17,)

    state, ts = jax.jit(env.reset)(jax.random.PRNGKey(0))
    assert int(ts.step_type) == int(StepType.FIRST)
    assert ts.observation.shape == (17,)

    state, ts = jax.jit(lambda s: env.step(s, jnp.asarray(0, dtype=jnp.int32)))(state)
    assert ts.observation.shape == (17,)
    assert ts.reward.shape == ()


def test_jax_reward_machine_matches_numpy_machine() -> None:
    """The JAX reward machine mirrors the NumPy PuckWorldRewardMachine exactly."""
    from examples.rm.anakin_puckworld.puckworld_machine import (
        JaxPuckWorldRewardMachine,
    )
    from examples.rm.discrete.core.machine import PuckWorldRewardMachine

    jax_delta_r = JaxPuckWorldRewardMachine()._get_reward_transition_function()
    numpy_delta_r = PuckWorldRewardMachine()._get_reward_transition_function()

    # Scalar target rewards are identical (e.g. +1000 on reaching T_3 at u=2).
    for state, expr in [(0, "T_1"), (1, "T_2"), (2, "T_3")]:
        assert jax_delta_r[state][expr] == numpy_delta_r[state][expr]

    # Shaping (NOT T_x) rewards agree numerically for a sampled observation.
    obs = np.zeros(dyn.GROUND_OBS_SIZE, dtype=np.float32)
    obs[0:2] = [0.5, 0.5]
    obs[4:6] = [0.9, 0.9]
    obs[6:8] = [-0.9, 0.9]
    obs[8:10] = [0.2, -0.3]
    jobs = jnp.asarray(obs)
    for state, expr in [(0, "NOT T_1"), (1, "NOT T_2"), (2, "NOT T_3")]:
        jax_val = float(jax_delta_r[state][expr](jobs, None, jobs))
        numpy_val = float(numpy_delta_r[state][expr](obs, None, obs))
        assert jax_val == pytest.approx(numpy_val, abs=1e-5)


def test_counterfactual_rows_have_fixed_shape() -> None:
    """Counterfactual experience yields one row per (machine state x counter cfg)."""
    from examples.rm.anakin_puckworld.puckworld_stoa_env import (
        make_puckworld_cross_product,
    )

    env = make_puckworld_cross_product(max_steps=50)
    ground = jnp.zeros((dyn.GROUND_OBS_SIZE,), dtype=jnp.float32)
    batch = env._core.generate_counterfactual_experience(
        ground, jnp.asarray(0, dtype=jnp.int32), ground
    )
    # PuckWorldRewardMachine has |U| machine states x |counter configs| rows,
    # each a 17-dim product observation.
    assert batch.obs.shape[1] == dyn.PRODUCT_OBS_SIZE
    assert batch.obs.shape[0] == batch.rewards.shape[0]


def test_anakin_dqn_smoke() -> None:
    """A few hundred Anakin steps run end-to-end and return metrics."""
    from examples.rm.anakin_puckworld.train_stoix_anakin import run_anakin_dqn

    metrics = run_anakin_dqn(total_timesteps=512, seed=0, log_dir="/tmp/pw_anakin",
                             num_envs=8)
    assert metrics["backend"].startswith("stoix") or metrics["backend"].startswith("anakin")
    assert metrics["steps_per_sec"] > 0
