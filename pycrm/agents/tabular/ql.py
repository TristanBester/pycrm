from collections import defaultdict

import gymnasium as gym
import numpy as np
from tqdm import tqdm


class QLearningAgent:
    """Q-Learning Agent."""

    def __init__(
        self,
        env: gym.Env,
        epsilon: float = 0.01,
        learning_rate: float = 0.01,
        discount_factor: float = 0.99,
    ) -> None:
        """Initialise the Q-Learning agent."""
        self.env = env
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))  # type: ignore

    def get_action(self, obs: np.ndarray) -> int:
        """Get the action with the largest Q-value."""
        return int(np.argmax(self.q_table[tuple(obs)]))

    def learn(self, total_episodes: int) -> np.ndarray:
        """Train the agent."""
        returns = []

        for _ in tqdm(range(total_episodes)):
            obs, _ = self.env.reset()
            done = False
            return_ = 0

            while not done:
                if np.random.random() < self.epsilon or np.all(
                    self.q_table[tuple(obs)] == 0
                ):
                    action = np.random.randint(0, self.env.action_space.n)  # type: ignore
                else:
                    action = self.get_action(obs)

                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                reward = float(reward)
                done = terminated or truncated

                if not done:
                    self.q_table[tuple(obs)][action] += self.learning_rate * (
                        reward
                        + self.discount_factor * np.max(self.q_table[tuple(next_obs)])
                        - self.q_table[tuple(obs)][action]
                    )
                else:
                    self.q_table[tuple(obs)][action] += self.learning_rate * (
                        reward - self.q_table[tuple(obs)][action]
                    )

                return_ += reward
                obs = next_obs

            returns.append(return_)
        return np.array(returns)
