import numpy as np
import pytest

pytest.importorskip("sbx")

from examples.jax.gridworld_dqn_convergence import (  # noqa: E402
    SequenceCrossProduct,
    run_benchmark,
)


def test_sequence_gridworld_counterfactuals_include_all_machine_states():
    """The deep gridworld exposes counterfactual transitions for DQN replay."""
    env = SequenceCrossProduct(max_steps=150)
    try:
        experiences = env.generate_counterfactual_experience(
            ground_obs=np.array([7, 0], dtype=np.int32),
            action=3,
            next_ground_obs=np.array([8, 0], dtype=np.int32),
        )
        cf_obs, cf_actions, cf_next_obs, cf_rewards, cf_dones, _ = experiences

        assert len(cf_obs) == 12
        assert cf_actions.tolist() == [3] * 12
        assert cf_next_obs.shape == cf_obs.shape
        assert cf_rewards.tolist() == [
            -0.01,
            -0.01,
            -0.01,
            0.5,
            -0.01,
            -0.01,
            -0.01,
            0.5,
            -0.01,
            -0.01,
            -0.01,
            3.0,
        ]
        assert cf_dones.tolist() == [False] * 11 + [True]
    finally:
        env.close()


def test_sequence_gridworld_sbx_dqn_smoke():
    """The convergence harness records training-return episodes."""
    result = run_benchmark(
        algorithms=["dqn"],
        backends=["sbx"],
        timesteps=64,
        seed=0,
        max_steps=16,
        buffer_size=512,
        batch_size=32,
        learning_starts=16,
        device="cpu",
    )

    run = result["runs"][0]
    assert run["backend"] == "sbx"
    assert run["algorithm"] == "dqn"
    assert run["training_curve"]
