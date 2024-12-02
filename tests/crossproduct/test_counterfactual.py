import numpy as np
import pytest

from crm.crossproduct.crossproduct import CrossProduct


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
