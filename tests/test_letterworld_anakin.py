import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
jax = pytest.importorskip("jax")
pytest.importorskip("stoa")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.introduction.core.ground import LetterWorld  # noqa: E402
from examples.rm.letterworld_anakin import letterworld_dynamics as dyn  # noqa: E402


def test_obs_layout_and_reset_match_numpy() -> None:
    """JAX reset obs is [symbol_seen=0, row=1, col=3], matching numpy LetterWorld."""
    state = dyn.reset_ground(jax.random.PRNGKey(0))
    obs = dyn.ground_obs(state)
    assert obs.shape == (dyn.GROUND_OBS_SIZE,)
    assert obs.tolist() == [0, 1, 3]


def test_movement_matches_numpy_for_all_actions() -> None:
    """Deterministic grid moves (with wall-clamping) match numpy LetterWorld."""
    numpy_env = LetterWorld()
    for start in ([1, 3], [0, 0], [2, 6], [1, 1]):
        for action in range(dyn.NUM_ACTIONS):
            numpy_env.agent_position = np.array(start)
            numpy_env.symbol_seen = True  # freeze the stochastic flip out of movement
            numpy_env._update_agent_position(action)
            state = dyn.LetterWorldGroundState(
                agent_pos=jnp.asarray(start, dtype=jnp.int32),
                symbol_seen=jnp.asarray(1, dtype=jnp.int32),
            )
            nxt = dyn.step_ground(state, jnp.asarray(action), jax.random.PRNGKey(0))
            assert nxt.agent_pos.tolist() == list(numpy_env.agent_position)


def test_symbol_flip_is_bernoulli_half_at_A_only() -> None:  # noqa: N802
    """At A with symbol unseen, flip ~ Bernoulli(0.5); never elsewhere or if seen."""
    at_a = dyn.LetterWorldGroundState(
        agent_pos=dyn.A_POSITION.astype(jnp.int32),
        symbol_seen=jnp.asarray(0, dtype=jnp.int32),
    )
    keys = jax.random.split(jax.random.PRNGKey(0), 2000)

    def flip_seen(state: dyn.LetterWorldGroundState) -> Any:
        def step(k: Any) -> Any:
            return dyn.step_ground(state, jnp.asarray(dyn.LEFT), k).symbol_seen

        return jax.vmap(step)(keys)

    # Moving LEFT off A ([1,1] -> [1,0]) means we do not arrive at A, so never flips.
    off_a = flip_seen(at_a)
    assert int(off_a.sum()) == 0
    # Arriving onto A from the right ([1,2] --LEFT--> [1,1]) with unseen -> ~50% flip.
    from_right = dyn.LetterWorldGroundState(
        agent_pos=jnp.asarray([1, 2], dtype=jnp.int32),
        symbol_seen=jnp.asarray(0, dtype=jnp.int32),
    )
    seen = flip_seen(from_right)
    rate = float(seen.mean())
    assert 0.4 < rate < 0.6
    # Already seen -> never changes.
    seen_state = dyn.LetterWorldGroundState(
        agent_pos=jnp.asarray([1, 2], dtype=jnp.int32),
        symbol_seen=jnp.asarray(1, dtype=jnp.int32),
    )
    stays = flip_seen(seen_state)
    assert int(stays.sum()) == len(keys)


def test_labels_vector_matches_labelling_semantics() -> None:
    """labels_vector returns [A, B, C] with the LetterWorld labelling semantics."""
    assert dyn.labels_vector(jnp.asarray([0, 1, 1])).tolist() == [True, False, False]
    assert dyn.labels_vector(jnp.asarray([1, 1, 1])).tolist() == [False, True, False]
    assert dyn.labels_vector(jnp.asarray([1, 1, 5])).tolist() == [False, False, True]
    assert dyn.labels_vector(jnp.asarray([0, 1, 5])).tolist() == [False, False, False]


def test_cross_product_shapes_and_first_timestep() -> None:
    """Product exposes a [ground + one_hot(u) + counter] obs and a FIRST timestep."""
    from stoa.env_types import StepType

    from examples.rm.letterworld_anakin.letterworld_stoa_env import (
        make_letterworld_cross_product,
    )

    env = make_letterworld_cross_product(max_steps=50)
    core = env._core
    expected_width = dyn.GROUND_OBS_SIZE + core.num_machine_states + core.num_counters
    assert env.observation_space().shape == (expected_width,)

    state, ts = jax.jit(env.reset)(jax.random.PRNGKey(0))
    assert int(ts.step_type) == int(StepType.FIRST)
    assert ts.observation.shape == (expected_width,)

    state, ts = jax.jit(lambda s: env.step(s, jnp.asarray(0, dtype=jnp.int32)))(state)
    assert ts.reward.shape == ()


def test_reward_parity_with_python_reward_machine() -> None:
    """Compiled-CRM reward matches the Python CRM: -0.1 per step, +1 on solve."""
    from examples.rm.letterworld_anakin.letterworld_stoa_env import (
        make_letterworld_cross_product,
    )

    env = make_letterworld_cross_product(max_steps=50)
    core = env._core
    # A transition that fires event C (seen=1 at C_POSITION): counterfactual rows cover
    # every (machine-state u, counter c). The (u=1, c=0) row is the terminal +1; all
    # other valid rows are the -0.1 shaping reward.
    prev = jnp.asarray([1, 1, 4], dtype=jnp.float32)  # seen, adjacent to C
    nxt = jnp.asarray([1, 1, 5], dtype=jnp.float32)  # seen, on C  -> event C
    batch = core.generate_counterfactual_experience(
        prev, jnp.asarray(dyn.RIGHT, dtype=jnp.int32), nxt
    )
    # The core enumerates counterfactual rows as the (machine-state x counter-config)
    # grid: each machine state is repeated over every counter config. Reconstruct the
    # per-row (u, c) grid the same way so we can locate individual rows.
    machine_states = np.asarray(core.counterfactual_machine_states)
    counter_configs = np.asarray(core.counterfactual_counter_configurations)
    us = np.repeat(machine_states, counter_configs.shape[0], axis=0)
    cs = np.tile(counter_configs, (machine_states.shape[0], 1))
    rewards = np.asarray(batch.rewards)
    dones = np.asarray(batch.dones)
    # locate the (u=1, counter=0) row
    idx = np.where((us == 1) & (cs[:, 0] == 0))[0]
    assert idx.size == 1
    assert rewards[idx[0]] == pytest.approx(1.0)
    assert bool(dones[idx[0]]) is True
    # a valid non-terminal row (e.g. u=0) carries the -0.1 shaping reward
    idx0 = np.where((us == 0) & (cs[:, 0] == 0))[0]
    assert rewards[idx0[0]] == pytest.approx(-0.1)


def test_anakin_dqn_smoke_and_schema() -> None:
    """A short Anakin run returns the shared metrics schema with eval curves."""
    from examples.rm.letterworld_anakin.train_anakin import run_anakin_dqn

    m = run_anakin_dqn(
        total_timesteps=4096, seed=0, log_dir="/tmp/lw_anakin", num_envs=8
    )
    assert m["backend"] == "anakin_dqn"
    for k in ("eval_steps", "eval_wall", "success_rate", "mean_return"):
        assert len(m[k]) == len(m["eval_steps"]) and len(m["eval_steps"]) >= 1
    assert m["steps_per_sec"] > 0
    assert all(np.isfinite(x) for x in m["mean_return"])
    assert 0.0 <= min(m["success_rate"]) and max(m["success_rate"]) <= 1.0
