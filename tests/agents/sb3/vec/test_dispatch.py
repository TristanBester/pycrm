import gymnasium as gym
import pytest

from crm.agents.sb3.vec import DispatchSubprocVecEnv


def mock_env_callable() -> gym.Env:
    """Create a mock environment that can be pickled."""

    class MockEnv(gym.Env):
        """A mock environment for testing purposes."""

        def __init__(self):
            super().__init__()
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,))
            self.action_space = gym.spaces.Discrete(2)

        def reset(
            self, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[float, dict]:
            """Reset the environment to an initial state."""
            return self.observation_space.sample(), {}

        def step(self, action) -> tuple[float, float, bool, bool, dict]:
            """Simulate taking a step in the environment."""
            return self.observation_space.sample(), 0, False, False, {}

        def increment_number(self, x: int) -> int:
            """A test method to demonstrate environment method dispatch."""
            return x + 1

    return MockEnv()


@pytest.fixture
def dispatch_env() -> DispatchSubprocVecEnv:
    """Fixture to create a DispatchSubprocVecEnv with mock environments."""
    return DispatchSubprocVecEnv(
        env_fns=[mock_env_callable, mock_env_callable, mock_env_callable]
    )


def test_dispatched_env_method(dispatch_env: DispatchSubprocVecEnv):
    """Test the dispatched environment method for correct functionality."""
    # Arguments to pass to the dispatched method
    args = (1, 2, 3)

    # Call the dispatched method on all environments
    results = dispatch_env.dispatched_env_method("increment_number", args)

    # Assert that the results are as expected
    assert len(results) == 3
    assert results == [2, 3, 4]
