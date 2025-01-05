import numpy as np
from tqdm import tqdm

from crm.agents.tabular.ql import QLearningAgent
from crm.crossproduct import CrossProduct


class CounterfactualQLearningAgent(QLearningAgent):
    """Counterfactual Q-Learning Agent."""

    def learn(self, total_episodes: int) -> np.ndarray:
        """Train the agent using counterfactual experience generation."""
        assert isinstance(self.env.unwrapped, CrossProduct)
        returns = []

        for _ in tqdm(range(total_episodes)):
            obs, _ = self.env.reset()
            done = False
            return_ = 0
            ep_len = 0

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

                for o, a, o_, r, d, _ in zip(
                    *self.env.unwrapped.generate_counterfactual_experience(
                        self.env.unwrapped.to_ground_obs(obs),
                        action,
                        self.env.unwrapped.to_ground_obs(next_obs),
                    ),
                    strict=True,
                ):
                    if not d:
                        self.q_table[tuple(o)][a] += self.learning_rate * (
                            r
                            + self.discount_factor * np.max(self.q_table[tuple(o_)])
                            - self.q_table[tuple(o)][a]
                        )
                    else:
                        self.q_table[tuple(o)][a] += self.learning_rate * (
                            r - self.q_table[tuple(o)][a]
                        )

                return_ += reward
                obs = next_obs
                ep_len += 1

            returns.append(return_)
        return np.array(returns)
