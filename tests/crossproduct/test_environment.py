from unittest.mock import MagicMock

import numpy as np

from tests.crossproduct.conftest import CrossProductMDP, DefaultCrossProduct


class TestCrossProductEnvironment:
    """Test the environment functionality of cross product class."""

    def test_reset(self, cross_product_mdp: CrossProductMDP) -> None:
        """Test the reset method."""
        obs, info = cross_product_mdp.reset()

        assert obs[0] == 0
        assert obs[1] == cross_product_mdp.crm.u_0
        assert obs[2] == cross_product_mdp.crm.c_0
        assert info == {}

    def test_step(self, cross_product_mdp: CrossProductMDP) -> None:
        """Test the step method."""
        cross_product_mdp.reset()
        obs, reward, terminated, truncated, info = cross_product_mdp.step(0)

        assert obs[0] == 1
        assert reward == 1
        assert not terminated
        assert not truncated
        assert info == {}

    def test_max_steps(self, cross_product_mdp: CrossProductMDP) -> None:
        """Test the max steps."""
        cross_product_mdp.reset()
        for _ in range(cross_product_mdp.max_steps - 1):
            _, _, _, truncated, _ = cross_product_mdp.step(0)
            assert not truncated

        _, _, _, truncated, _ = cross_product_mdp.step(0)
        assert truncated

    def test_render(self, cross_product_mdp: CrossProductMDP) -> None:
        """Test the render method."""
        cross_product_mdp.ground_env = MagicMock()
        cross_product_mdp.render()
        cross_product_mdp.ground_env.render.assert_called_once()


class TestDefaultObservation:
    """Test the default _get_obs and to_ground_obs implementations."""

    def test_get_obs_structure(self, default_cross_product: DefaultCrossProduct) -> None:
        """Test that default _get_obs concatenates ground obs, one-hot u, and raw c."""
        ground_obs = np.array([5.0, 6.0], dtype=np.float32)
        u = 0
        c = (3,)

        obs = default_cross_product._get_obs(ground_obs, u, c)

        u_size = len(default_cross_product.crm.U) + len(default_cross_product.crm.F)
        c_size = len(c)
        expected_len = len(ground_obs) + u_size + c_size
        assert len(obs) == expected_len

    def test_get_obs_one_hot_encoding(self, default_cross_product: DefaultCrossProduct) -> None:
        """Test that u is one-hot encoded over all machine states."""
        ground_obs = np.array([0.0], dtype=np.float32)
        crm = default_cross_product.crm
        u_size = len(crm.U) + len(crm.F)

        for u in crm.U:
            obs = default_cross_product._get_obs(ground_obs, u, crm.c_0)
            u_enc = obs[len(ground_obs) : len(ground_obs) + u_size]
            assert u_enc[u] == 1.0
            assert np.sum(u_enc) == 1.0

    def test_get_obs_counter_values(self, default_cross_product: DefaultCrossProduct) -> None:
        """Test that counter values are passed through raw."""
        ground_obs = np.array([0.0], dtype=np.float32)
        c = (7,)

        obs = default_cross_product._get_obs(ground_obs, 0, c)

        assert obs[-1] == 7.0

    def test_get_obs_ground_obs_preserved(self, default_cross_product: DefaultCrossProduct) -> None:
        """Test that ground observation is preserved at the start."""
        ground_obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        obs = default_cross_product._get_obs(ground_obs, 0, (0,))

        np.testing.assert_array_equal(obs[:3], ground_obs)

    def test_to_ground_obs(self, default_cross_product: DefaultCrossProduct) -> None:
        """Test that to_ground_obs inverts _get_obs."""
        ground_obs = np.array([0.0], dtype=np.float32)
        u = 0
        c = default_cross_product.crm.c_0

        obs = default_cross_product._get_obs(ground_obs, u, c)
        recovered = default_cross_product.to_ground_obs(obs)

        np.testing.assert_array_equal(recovered, ground_obs)

    def test_to_ground_obs_after_step(self, default_cross_product: DefaultCrossProduct) -> None:
        """Test that to_ground_obs recovers ground obs from a stepped observation."""
        default_cross_product.reset()
        obs, _, _, _, _ = default_cross_product.step(0)

        recovered = default_cross_product.to_ground_obs(obs)

        # GroundEnv.step returns [1]
        np.testing.assert_array_equal(recovered, np.array([1.0]))
