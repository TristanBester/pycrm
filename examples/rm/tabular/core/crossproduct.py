import gymnasium as gym
import numpy as np

from pycrm.automaton import RewardMachine
from pycrm.crossproduct import CrossProduct
from pycrm.label import LabellingFunction


class OfficeWorldCrossProduct(CrossProduct[np.ndarray, np.ndarray, int, None]):
    """Cross product of the Office World environment."""

    def __init__(
        self,
        ground_env: gym.Env,
        machine: RewardMachine,
        lf: LabellingFunction[np.ndarray, int],
        max_steps: int,
    ) -> None:
        """Initialize the cross product Markov decision process environment."""
        super().__init__(ground_env, machine, lf, max_steps)
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
        del c
        return np.array([ground_obs[0], ground_obs[1], ground_obs[2], u])

    def to_ground_obs(self, obs: np.ndarray) -> np.ndarray:
        """Convert the cross product observation to a ground observation.

        Args:
            obs: The cross product observation.

        Returns:
            Ground observation - [agent row, agent col, mail empty].
        """
        return obs[:3]
