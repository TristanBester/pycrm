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
        memory_scale: float = 1.0,
    ) -> None:
        """Initialize the cross product MDP environment."""
        super().__init__(ground_env, crm, lf, max_steps)

        n_counters = len(crm.c_0)
        self.observation_space = gym.spaces.Box(
            low=0, high=100, shape=(7 + 1 + n_counters,), dtype=np.float32
        )
        self.action_space = self.ground_env.action_space
        self.render_mode = self.ground_env.render_mode
        self.metadata = self.ground_env.metadata
        self.memory_scale = memory_scale

    def _get_obs(
        self, ground_obs: np.ndarray, u: int, c: tuple[int, ...]
    ) -> np.ndarray:
        """Get the cross product observation.

        Args:
            ground_obs: The ground observation.
            u: The number of symbols seen.
            c: The counter configuration.
        """
        machine_state = np.array([u])
        memory = np.array(c) / self.memory_scale
        configuration = np.concatenate([machine_state, memory])
        return np.concatenate([ground_obs, configuration])

    def to_ground_obs(self, obs: np.ndarray) -> np.ndarray:
        """Convert the cross product observation to a ground observation.

        Args:
            obs: The cross product observation.

        Returns:
            Ground observation - [agent_position].
        """
        return obs[:7]
