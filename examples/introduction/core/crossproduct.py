import gymnasium as gym
import numpy as np

from pycrm.automaton import CountingRewardMachine
from pycrm.crossproduct import CrossProduct
from pycrm.label import LabellingFunction


class LetterWorldCrossProduct(CrossProduct[np.ndarray, np.ndarray, int, None]):
    """Cross product of the Letter World environment."""

    def __init__(
        self,
        ground_env: gym.Env,
        crm: CountingRewardMachine,
        lf: LabellingFunction[np.ndarray, int],
        max_steps: int,
    ) -> None:
        """Initialize the cross product Markov decision process environment."""
        super().__init__(ground_env, crm, lf, max_steps)
        self.observation_space = gym.spaces.Box(
            low=0, high=100, shape=(3,), dtype=np.int32
        )
        self.action_space = self.ground_env.action_space

    def _get_obs(
        self, ground_obs: np.ndarray, u: int, c: tuple[int, ...]
    ) -> np.ndarray:
        """Get the cross product observation.

        Args:
            ground_obs: The ground observation.
            u: The number of symbols seen.
            c: The counter configuration.

        Returns:
            Cross product observation - [agent_position, machine state, counter state].
        """
        return np.array([ground_obs[0], ground_obs[1], ground_obs[2], u, c[0]])

    def to_ground_obs(self, obs: np.ndarray) -> np.ndarray:
        """Convert the cross product observation to a ground observation.

        Args:
            obs: The cross product observation.

        Returns:
            Ground observation - [symbol_seen, agent_row, agent_col].
        """
        return obs[:3]
