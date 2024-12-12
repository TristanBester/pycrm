import gymnasium as gym
import numpy as np

from crm.automaton import CountingRewardMachine
from crm.crossproduct import CrossProduct
from crm.label import LabellingFunction


class WarehouseCrossProduct(CrossProduct[np.ndarray, np.ndarray, np.ndarray, None]):
    """Cross product MDP of the warehouse environment."""

    def __init__(
        self,
        ground_env: gym.Env,
        crm: CountingRewardMachine,
        lf: LabellingFunction[np.ndarray, np.ndarray],
        max_steps: int = 400,
    ) -> None:
        """Initialize the cross product MDP environment."""
        super().__init__(ground_env, crm, lf, max_steps)

        # TODO: The counter machine must make it easy to define the observation space
        # of the cross product MDP. (like expose machine state + counters shape)
        self.observation_space = gym.spaces.Box(
            low=0, high=100, shape=(7 + 2,), dtype=np.float32
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
        """
        return np.concatenate([ground_obs, np.array([u, c[0]])])

    def _to_ground_obs(self, obs: np.ndarray) -> np.ndarray:
        """Convert the cross product observation to a ground observation.

        Args:
            obs: The cross product observation.

        Returns:
            Ground observation - [agent_position].
        """
        return obs[:7]
