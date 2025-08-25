import numpy as np
import pytest

from pycrm.crossproduct.crossproduct import CrossProduct


@pytest.fixture
def counterfactual_experiences(
    cross_product_mdp: CrossProduct,
) -> tuple[np.ndarray, ...]:
    """Return counterfactual experiences."""
    return cross_product_mdp.generate_counterfactual_experience(
        ground_obs=np.array([0]),
        action=0,
        next_ground_obs=np.array([1]),
    )


class TestCounterfactualExperiences:
    """Test counterfactual experience generation."""

    def test_counterfactual_observations(
        self, counterfactual_experiences: tuple[np.ndarray, ...]
    ) -> None:
        """Test the counterfactual observations."""
        obs_buffer = counterfactual_experiences[0]

        # Test ground environment observation
        assert np.all(obs_buffer[:, 0] == 0)

        # Test machine state
        assert np.all(obs_buffer[:3, 1] == 0)
        assert np.all(obs_buffer[3:, 1] == 1)

        # Test counter configuration
        assert np.all(obs_buffer[[0, 3], 2] == 0)
        assert np.all(obs_buffer[[1, 4], 2] == 1)
        assert np.all(obs_buffer[[2, 5], 2] == 2)

    def test_counterfactual_actions(
        self, counterfactual_experiences: tuple[np.ndarray, ...]
    ) -> None:
        """Test the counterfactual actions."""
        action_buffer = counterfactual_experiences[1]
        assert np.all(action_buffer == 0)

    def test_counterfactual_next_observations(
        self, counterfactual_experiences: tuple[np.ndarray, ...]
    ) -> None:
        """Test the counterfactual next observations."""
        next_obs_buffer = counterfactual_experiences[2]

        # Test ground environment observation
        assert np.all(next_obs_buffer[:, 0] == 1)

        # Test machine state
        assert np.all(next_obs_buffer[:3, 1] == 1)
        assert np.all(next_obs_buffer[3:, 1] == 2)

        # Test counter configuration
        assert next_obs_buffer[0, 2] == 1
        assert next_obs_buffer[1, 2] == 2
        assert next_obs_buffer[2, 2] == 3
        assert next_obs_buffer[3, 2] == 0
        assert next_obs_buffer[4, 2] == 1
        assert next_obs_buffer[5, 2] == 2

    def test_counterfactual_rewards(
        self, counterfactual_experiences: tuple[np.ndarray, ...]
    ) -> None:
        """Test the counterfactual rewards."""
        reward_buffer = counterfactual_experiences[3]
        assert np.all(reward_buffer[:3] == 1.0)
        assert np.all(reward_buffer[3:] == 0.0)

    def test_counterfactual_dones(
        self, counterfactual_experiences: tuple[np.ndarray, ...]
    ) -> None:
        """Test the counterfactual dones."""
        done_buffer = counterfactual_experiences[4]
        assert np.all(~done_buffer[:3])
        assert np.all(done_buffer[3:])

    def test_counterfactual_infos(
        self, counterfactual_experiences: tuple[np.ndarray, ...]
    ) -> None:
        """Test the counterfactual infos."""
        info_buffer = counterfactual_experiences[5]
        assert np.all(info_buffer == {})


class TestCounterfactualWithAdapter:
    """Test counterfactual experience generation with RmToCrmAdapter."""

    def test_rm_to_crm_adapter_creation(self):
        """Test that CrossProduct correctly creates RmToCrmAdapter for RewardMachine."""
        from pycrm.automaton import RmToCrmAdapter
        from tests.crossproduct.conftest import (
            RM,
            CrossProductMDP,
            Events,
            GroundEnv,
            LabelFunction,
        )

        # Create a RewardMachine
        rm = RM(env_prop_enum=Events)

        # Create CrossProduct with RewardMachine - this should trigger line 32
        ground_env = GroundEnv()
        labelling_function = LabelFunction()

        # This should create an RmToCrmAdapter internally
        cross_product = CrossProductMDP(
            ground_env=ground_env,
            machine=rm,  # This is a RewardMachine, not CountingRewardMachine
            lf=labelling_function,
            max_steps=10,
        )

        # Verify that the internal crm is an RmToCrmAdapter
        assert hasattr(cross_product, "crm")
        assert isinstance(cross_product.crm, RmToCrmAdapter)
        assert cross_product.crm._rm is rm

    def test_counterfactual_with_reward_machine_adapter(
        self, cross_product_mdp_with_adapter: CrossProduct
    ) -> None:
        """Test counterfactual experience generation using a RewardMachine."""
        counterfactual_experiences = (
            cross_product_mdp_with_adapter.generate_counterfactual_experience(
                ground_obs=np.array([0]),
                action=0,
                next_ground_obs=np.array([1]),
            )
        )
        obs_buffer = counterfactual_experiences[0]

        # Test ground environment observation
        assert np.all(obs_buffer[:, 0] == 0)

        # Test machine state - should have states from the reward machine (0, 1, 2)
        # The adapter should iterate through all states from the RM
        assert len(obs_buffer) > 0  # Should have some observations

        # Test counter configuration - should always be (0,) for RM adapter
        assert np.all(obs_buffer[:, 2] == 0)


class TestCounterfactualErrorHandling:
    """Test error handling in counterfactual experience generation."""

    def test_counterfactual_transition_error_handling(self):
        """Test that ValueError in transition is handled correctly."""
        # This test covers lines 109-111 in crossproduct.py
        # (ValueError exception handling)
        from pycrm.automaton import CountingRewardMachine
        from tests.crossproduct.conftest import (
            CrossProductMDP,
            Events,
            GroundEnv,
            LabelFunction,
        )

        # Create a custom CRM that will definitely trigger ValueError
        class FailingCRM(CountingRewardMachine):
            """A CRM that always raises ValueError for certain transitions."""

            def __init__(self):
                super().__init__(env_prop_enum=Events)

            @property
            def u_0(self) -> int:
                return 0

            @property
            def c_0(self) -> tuple[int, ...]:
                return (0,)

            def _get_state_transition_function(self) -> dict:
                return {
                    0: {
                        # No transitions defined for state 0 - this will
                        # cause ValueError
                    },
                    1: {
                        "EVENT_A / (-)": 2,
                    },
                }

            def _get_counter_transition_function(self) -> dict:
                return {
                    0: {
                        # No counter transitions for state 0
                    },
                    1: {
                        "EVENT_A / (-)": (0,),
                    },
                }

            def _get_reward_transition_function(self) -> dict:
                return {
                    0: {
                        # No reward transitions for state 0
                    },
                    1: {
                        "EVENT_A / (-)": 1.0,
                    },
                }

            def sample_counter_configurations(self) -> list[tuple[int, ...]]:
                return [(0,)]

        # Create the failing CRM
        crm = FailingCRM()

        # Create CrossProduct with this CRM
        ground_env = GroundEnv()
        labelling_function = LabelFunction()
        cross_product = CrossProductMDP(
            ground_env=ground_env,
            machine=crm,
            lf=labelling_function,
            max_steps=10,
        )

        # This should trigger the ValueError handling in
        # generate_counterfactual_experience
        # because state 0 has no transitions defined
        counterfactual_experiences = cross_product.generate_counterfactual_experience(
            ground_obs=np.array([0]),
            action=0,
            next_ground_obs=np.array([1]),
        )

        # The method should complete successfully, handling the ValueError internally
        assert len(counterfactual_experiences) == 6
        # Since all transitions from state 0 fail, we might get fewer experiences
        # but the important thing is that no exception was raised
