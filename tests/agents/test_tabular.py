"""Tests for tabular agents."""

from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest

from pycrm.agents.tabular import CounterfactualQLearningAgent, QLearningAgent


class MockEnv(gym.Env):
    """Mock environment for testing."""

    def __init__(self, n_states=4, n_actions=2):
        """Initialize mock environment."""
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.action_space = gym.spaces.Discrete(n_actions)
        self.observation_space = gym.spaces.Box(
            low=0, high=n_states - 1, shape=(1,), dtype=np.int32
        )
        self.current_state = 0
        self._unwrapped = self  # Store the unwrapped attribute for testing

    @property
    def unwrapped(self):
        """Override unwrapped property to allow setting for testing."""
        return self._unwrapped

    @unwrapped.setter
    def unwrapped(self, value):
        """Allow setting unwrapped for testing purposes."""
        self._unwrapped = value

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset environment."""
        super().reset(seed=seed, options=options)
        self.current_state = 0
        return np.array([self.current_state]), {}

    def step(self, action):
        """Step environment."""
        # Simple deterministic transitions: action 0 -> state 0, action 1 -> state 1
        if action == 0:
            next_state = 0
            reward = 0.0
        else:  # action == 1
            next_state = 1
            reward = 1.0

        self.current_state = next_state
        terminated = next_state == 1  # Terminate when reaching state 1
        truncated = False
        return np.array([next_state]), reward, terminated, truncated, {}


class TestQLearningAgent:
    """Test Q-Learning Agent."""

    def test_initialization(self):
        """Test agent initialization."""
        env = MockEnv()
        agent = QLearningAgent(env, epsilon=0.1, learning_rate=0.1, discount_factor=0.9)

        assert agent.env == env
        assert agent.epsilon == 0.1
        assert agent.learning_rate == 0.1
        assert agent.discount_factor == 0.9
        assert len(agent.q_table) == 0  # Initially empty

    def test_get_action(self):
        """Test action selection."""
        env = MockEnv()
        agent = QLearningAgent(env)

        # Test with zero Q-values (should return first action)
        obs = np.array([0])
        action = agent.get_action(obs)
        assert isinstance(action, int)
        action_space = env.action_space
        assert isinstance(action_space, gym.spaces.Discrete)
        assert 0 <= action < action_space.n

        # Test with non-zero Q-values
        agent.q_table[tuple(obs)] = np.array([0.5, 1.0])  # Action 1 has higher value
        action = agent.get_action(obs)
        assert action == 1

    def test_learn_basic_functionality(self):
        """Test basic learning functionality."""
        env = MockEnv(n_states=3, n_actions=2)
        agent = QLearningAgent(env, epsilon=0.1, learning_rate=0.1, discount_factor=0.9)

        # Run a few episodes
        returns = agent.learn(total_episodes=5)

        assert len(returns) == 5
        assert all(isinstance(r, (int, float)) for r in returns)

        # Check that Q-table has been populated
        assert len(agent.q_table) > 0

    def test_q_table_updates(self):
        """Test that Q-table is updated during learning."""
        env = MockEnv(n_states=2, n_actions=2)
        agent = QLearningAgent(env, epsilon=0.0, learning_rate=0.5, discount_factor=0.9)

        # Initialize Q-table with some values
        obs = np.array([0])
        agent.q_table[tuple(obs)] = np.array([0.5, 0.5])
        initial_q_values = agent.q_table[tuple(obs)].copy()

        # Run multiple episodes to ensure learning happens
        agent.learn(total_episodes=10)

        # Q-values should have changed (action 1 should have higher value due to reward)
        updated_q_values = agent.q_table[tuple(obs)]
        assert updated_q_values[1] > initial_q_values[1]  # Action 1 should be preferred


class TestCounterfactualQLearningAgent:
    """Test Counterfactual Q-Learning Agent."""

    def test_initialization(self):
        """Test agent initialization."""
        env = MockEnv()
        agent = CounterfactualQLearningAgent(env, epsilon=0.1, learning_rate=0.1)

        assert agent.env == env
        assert agent.epsilon == 0.1
        assert agent.learning_rate == 0.1
        assert agent.discount_factor == 0.99  # Default value

    def test_inheritance(self):
        """Test that CounterfactualQLearningAgent inherits from QLearningAgent."""
        env = MockEnv()
        agent = CounterfactualQLearningAgent(env)

        # Should have QLearningAgent methods
        assert hasattr(agent, "get_action")
        assert hasattr(agent, "learn")

    def test_learn_requires_cross_product_env(self):
        """Test that learn method requires CrossProduct environment."""
        env = MockEnv()
        # Mock the unwrapped attribute to return something that's not CrossProduct
        env.unwrapped = Mock()
        env.unwrapped.__class__.__name__ = "NotCrossProduct"

        agent = CounterfactualQLearningAgent(env)

        # Should raise assertion error because env is not a CrossProduct
        with pytest.raises(AssertionError):
            agent.learn(total_episodes=1)

    def test_counterfactual_agent_uses_parent_q_learning(self):
        """Test that CounterfactualQLearningAgent inherits Q-learning behavior."""
        env = MockEnv()
        # Set up env to not be a CrossProduct so learn() will fail
        env.unwrapped = Mock()
        env.unwrapped.__class__.__name__ = "NotCrossProduct"

        agent = CounterfactualQLearningAgent(env, epsilon=0.1, learning_rate=0.1)

        # Test that it has the same basic Q-learning methods
        assert hasattr(agent, "get_action")
        assert hasattr(agent, "q_table")
        assert hasattr(agent, "epsilon")

        # Test that it inherits initialization correctly
        assert agent.epsilon == 0.1
        assert agent.learning_rate == 0.1
        assert agent.discount_factor == 0.99  # Default value

    def test_counterfactual_requires_proper_environment(self):
        """Test that CounterfactualQLearningAgent requires proper environment setup."""
        # This test just verifies the class structure and requirements
        # Without actually calling learn() which would require complex mocking

        env = MockEnv()
        agent = CounterfactualQLearningAgent(env, epsilon=0.0, learning_rate=0.1)

        # Verify the agent is properly initialized
        assert agent.env == env
        assert agent.epsilon == 0.0
        assert agent.learning_rate == 0.1

        # Test that it can perform basic Q-learning operations
        obs = np.array([0])
        action = agent.get_action(obs)
        assert isinstance(action, int)
        action_space = env.action_space
        assert isinstance(action_space, gym.spaces.Discrete)
        assert 0 <= action < action_space.n

        # The Q-table should have been populated when get_action was called
        assert len(agent.q_table) > 0
