import gymnasium as gym
import numpy as np

from pycrm.automaton import CountingRewardMachine, RewardMachine
from pycrm.crossproduct import CrossProduct
from pycrm.label import LabellingFunction


class PuckWorldCrossProduct(CrossProduct[np.ndarray, np.ndarray, np.ndarray, None]):
    """Cross product of the Puck World environment."""

    def __init__(
        self,
        ground_env: gym.Env,
        machine: CountingRewardMachine | RewardMachine,
        lf: LabellingFunction[np.ndarray, np.ndarray],
        max_steps: int,
    ) -> None:
        """Initialize the cross product Markov decision process environment."""
        super().__init__(ground_env, machine, lf, max_steps)
        self.observation_space = gym.spaces.Box(
            low=0, high=10, shape=(17,), dtype=np.float32
        )
        self.action_space = self.ground_env.action_space

    def _get_obs(
        self, ground_obs: np.ndarray, u: int, c: tuple[int, ...]
    ) -> np.ndarray:
        """Get the cross product observation.

        Args:
            ground_obs: The ground observation.z
            u: The number of symbols seen.
            c: The counter configuration.

        Returns:
            Cross product observation - [ground obs, machine state, counter state].
        """
        u_enc = np.zeros(len(self.crm.U) + 1, dtype=np.float32)
        u_enc[u] = 1
        crm_cfg = u_enc
        return np.concatenate((ground_obs, crm_cfg, c), axis=0)

    def to_ground_obs(self, obs: np.ndarray) -> np.ndarray:
        """Convert the cross product observation to a ground observation.

        Args:
            obs: The cross product observation.

        Returns:
            Ground observation.
        """
        return obs[:12]
