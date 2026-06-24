from typing import Any, cast

import pytest

pytest.importorskip("sbx")


def test_sbx_dqn_uses_sb3_parity_kwargs():
    """The JAX DQN benchmark path does not tune SBX-only kwargs."""
    from examples.jax.function_approximation_speed import (
        algorithm_kwargs,
        load_algorithm,
        make_task_env,
    )
    from pycrm.agents.sbx.dqn import DQN, CounterfactualDQN

    env = make_task_env("rm-discrete", 1000, 123)
    try:
        dqn_kwargs = algorithm_kwargs(
            task_name="rm-discrete",
            algorithm="dqn",
            backend="sbx",
            env=env,
            buffer_size=1_000_000,
            batch_size=2_500,
            learning_starts=100,
            device="cpu",
            seed=123,
        )
        cdqn_kwargs = algorithm_kwargs(
            task_name="rm-discrete",
            algorithm="cdqn",
            backend="sbx",
            env=env,
            buffer_size=1_000_000,
            batch_size=2_500,
            learning_starts=100,
            device="cpu",
            seed=123,
        )

        assert load_algorithm("sbx", "dqn") is DQN
        assert load_algorithm("sbx", "cdqn") is CounterfactualDQN
        assert issubclass(CounterfactualDQN, DQN)
        for kwargs in (dqn_kwargs, cdqn_kwargs):
            assert "learning_rate" not in kwargs
            assert kwargs["batch_size"] == 2_500
            assert kwargs["exploration_final_eps"] == 0.1
            assert "target_update_interval" not in kwargs
        assert dqn_kwargs["exploration_fraction"] == 0.1
        assert cdqn_kwargs["exploration_fraction"] == 0.2
    finally:
        env.close()


def test_sbx_and_sb3_counterfactual_replay_store_same_transition():
    """SB3 and SBX C-DQN insert the same counterfactual replay rows."""
    import numpy as np

    from examples.rm.discrete.jax_validation import make_env
    from pycrm.agents.sb3.dqn import CounterfactualDQN as Sb3CounterfactualDQN
    from pycrm.agents.sbx.dqn import CounterfactualDQN as SbxCounterfactualDQN

    def store_one_transition(model_class):
        np.random.seed(123)
        env = make_env(seed=123, max_steps=1000)
        model = model_class(
            "MlpPolicy",
            env,
            buffer_size=100,
            batch_size=32,
            learning_starts=100,
            learning_rate=1e-4,
            exploration_fraction=0.2,
            exploration_final_eps=0.1,
            target_update_interval=10_000,
            policy_kwargs={"net_arch": [64, 64]},
            device="cpu",
            seed=123,
        )
        try:
            model.env.seed(123)
            obs = model.env.reset()
            model._last_obs = obs
            actions = np.array([0])
            next_obs, _, dones, infos = model.env.step(actions)
            model._store_counterfactual_transitions(
                model.replay_buffer,
                actions,
                next_obs,
                dones,
                infos,
            )
            count = int(model.replay_buffer.pos)
            return {
                "observations": model.replay_buffer.observations[:count].copy(),
                "next_observations": model.replay_buffer.next_observations[
                    :count
                ].copy(),
                "actions": model.replay_buffer.actions[:count].copy(),
                "rewards": model.replay_buffer.rewards[:count].copy(),
                "dones": model.replay_buffer.dones[:count].copy(),
                "timeouts": model.replay_buffer.timeouts[:count].copy(),
            }
        finally:
            model.env.close()

    sb3_rows = store_one_transition(Sb3CounterfactualDQN)
    sbx_rows = store_one_transition(SbxCounterfactualDQN)

    assert sb3_rows.keys() == sbx_rows.keys()
    for key in sb3_rows:
        np.testing.assert_allclose(sbx_rows[key], sb3_rows[key])


def test_sbx_dqn_wrapper_uses_sb3_optimizer_semantics():
    """The SBX DQN wrapper applies SB3's DQN defaults and gradient clipping."""
    from examples.rm.discrete.jax_validation import make_env
    from pycrm.agents.sbx.dqn import DQN

    env = make_env(seed=123, max_steps=1000)
    model = DQN(
        "MlpPolicy",
        env,
        buffer_size=100,
        learning_starts=100,
        device="cpu",
        seed=123,
    )
    try:
        assert model.learning_rate == 1e-4
        assert model.batch_size == 32
        assert model.exploration_fraction == 0.1
        assert model.exploration_final_eps == 0.05
        assert model.target_update_interval == 10_000

        optimizer_class = cast(Any, model.policy.optimizer_class)
        assert optimizer_class.func.__name__ == "_clipped_adam"
        assert optimizer_class.keywords == {"max_grad_norm": 10.0}
    finally:
        env.close()
