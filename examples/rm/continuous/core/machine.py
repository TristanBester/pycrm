from typing import Callable

import numpy as np

from crm.automaton import RewardMachine
from examples.rm.continuous.core.label import Symbol


class PuckWorldRewardMachine(RewardMachine):
    """Reward machine for the Puck World environment."""

    def __init__(self):
        """Initialise the counting reward machine."""
        super().__init__(env_prop_enum=Symbol)

    @property
    def u_0(self) -> int:
        """Return the initial state of the machine."""
        return 0

    @property
    def encoded_configuration_size(self) -> int:
        """Return the size of the encoded counter configuration."""
        return 4

    def _get_state_transition_function(self) -> dict:
        """Return the state transition function."""
        return {
            0: {
                "T_1": 1,
                "DEFAULT": 0,
            },
            1: {
                "T_2": 2,
                "DEFAULT": 1,
            },
            2: {
                "T_3": 3,
                "DEFAULT": 2,
            },
            3: {
                "T_1": 4,
                "DEFAULT": 3,
            },
            4: {
                "T_2": 5,
                "DEFAULT": 4,
            },
            5: {
                "T_3": 6,
                "DEFAULT": 5,
            },
            6: {
                "T_1": 7,
                "DEFAULT": 6,
            },
            7: {
                "T_2": 8,
                "DEFAULT": 7,
            },
            8: {
                "T_3": -1,
                "DEFAULT": 8,
            },
        }

    def _get_reward_transition_function(self) -> dict:
        """Return the reward transition function."""
        return {
            0: {
                "T_1": 10,
                "DEFAULT": self._create_nav_t_1_reward(),
            },
            1: {
                "T_2": 10,
                "DEFAULT": self._create_nav_t_2_reward(),
            },
            2: {
                "T_3": 10,
                "DEFAULT": self._create_nav_t_3_reward(),
            },
            3: {
                "T_1": 10,
                "DEFAULT": self._create_nav_t_1_reward(),
            },
            4: {
                "T_2": 10,
                "DEFAULT": self._create_nav_t_2_reward(),
            },
            5: {
                "T_3": 10,
                "DEFAULT": self._create_nav_t_3_reward(),
            },
            6: {
                "T_1": 10,
                "DEFAULT": self._create_nav_t_1_reward(),
            },
            7: {
                "T_2": 10,
                "DEFAULT": self._create_nav_t_2_reward(),
            },
            8: {
                "T_3": 10,
                "DEFAULT": self._create_nav_t_3_reward(),
            },
        }

    def _create_nav_t_1_reward(
        self,
    ) -> Callable[[np.ndarray, np.ndarray, np.ndarray], float]:
        """Create the reward function for navigating to target 1."""

        def nav_t_1_reward(
            obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
        ) -> float:
            del obs, action

            agent_pos = next_obs[:2]
            target_one_pos = next_obs[4:6]
            dist = float(np.linalg.norm(agent_pos - target_one_pos))
            return -dist - 10

        return nav_t_1_reward

    def _create_nav_t_2_reward(
        self,
    ) -> Callable[[np.ndarray, np.ndarray, np.ndarray], float]:
        """Create the reward function for navigating to target 2."""

        def nav_t_2_reward(
            obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
        ) -> float:
            del obs, action

            agent_pos = next_obs[:2]
            target_two_pos = next_obs[6:8]
            dist = float(np.linalg.norm(agent_pos - target_two_pos))
            return -dist - 5

        return nav_t_2_reward

    def _create_nav_t_3_reward(
        self,
    ) -> Callable[[np.ndarray, np.ndarray, np.ndarray], float]:
        """Create the reward function for navigating to target 3."""

        def nav_t_3_reward(
            obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
        ) -> float:
            del obs, action

            agent_pos = next_obs[:2]
            target_three_pos = next_obs[8:10]
            dist = float(np.linalg.norm(agent_pos - target_three_pos))
            return -dist

        return nav_t_3_reward
