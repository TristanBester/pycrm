import gymnasium as gym
import numpy as np


class EEStateWrapper(gym.ObservationWrapper):
    """Wraps the environment to return the end-effector state."""

    def __init__(self, env: gym.Env[dict[str, np.ndarray], np.ndarray]) -> None:
        """Initialise the wrapper."""
        super().__init__(env=env)
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0]),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1]),
        )
        self.action_space = env.action_space

    def observation(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        """Returns the state of the end effector as the observation.

        Returns:
            np.ndarray: The observation. The first 3 elements are the end-effector
                position, the next 3 are the end-effector velocity, and the last
                element is the gripper width.
        """
        return observation["observation"][:7]
