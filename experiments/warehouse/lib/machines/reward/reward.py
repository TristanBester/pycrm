from typing import Callable

import numpy as np


class RewardFunction:
    """Interface for reward functions with automatic rescaling capabilities."""

    def __init__(
        self,
        reward_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
        min_reward: float,
        max_reward: float,
        intercept: float = 0.0,
        enable_rescaling: bool = True,
    ) -> None:
        """Initialize the reward function interface.

        Args:
            reward_fn: Callable that takes (obs, action, next_obs) arrays and
                returns float
            min_reward: Minimum possible reward value
            max_reward: Maximum possible reward value
            enable_rescaling: Whether to rescale rewards to [0,1] range
            intercept: Intercept to subtract from the scaled reward
        Raises:
            ValueError: If min_reward is greater than or equal to max_reward
        """
        if min_reward > max_reward:
            raise ValueError(
                f"min_reward ({min_reward}) must be less than max_reward ({max_reward})"
            )

        self._reward_fn = reward_fn
        self._enable_rescaling = enable_rescaling
        self._min_reward = min_reward
        self._max_reward = max_reward
        self._intercept = intercept

    def _scale_reward(self, reward: float) -> float:
        """Scale reward to [0,1] range using defined bounds."""
        return (reward - self._min_reward) / (self._max_reward - self._min_reward)

    def __call__(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> float:
        """Call the reward function and apply rescaling if enabled.

        Args:
            obs: Current obs array
            action: Action array
            next_obs: Next obs array

        Returns:
            Scaled reward value between 0 and 1 if rescaling is enabled,
            otherwise returns the original reward value.
        """
        reward = self._reward_fn(obs, action, next_obs)
        clipped_reward = np.clip(reward, self._min_reward, self._max_reward)

        if not self._enable_rescaling:
            return clipped_reward + self._intercept

        scaled_reward = self._scale_reward(clipped_reward)
        return scaled_reward + self._intercept
