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
            low=0, high=100, shape=(17,), dtype=np.float32
        )
        self.action_space = self.ground_env.action_space
