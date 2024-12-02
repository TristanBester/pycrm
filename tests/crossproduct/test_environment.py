from unittest.mock import MagicMock

from tests.crossproduct.conftest import CrossProductMDP


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
