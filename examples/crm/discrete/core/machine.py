from typing import Callable

import numpy as np

from examples.crm.discrete.core.label import Symbol
from pycrm.automaton import CountingRewardMachine


class PuckWorldCountingRewardMachine(CountingRewardMachine):
    """Counting reward machine for the Puck World environment."""

    def __init__(self):
        """Initialise the counting reward machine."""
        super().__init__(env_prop_enum=Symbol)

    @property
    def u_0(self) -> int:
        """Return the initial state of the machine."""
        return 0

    @property
    def c_0(self) -> tuple[int, ...]:
        """Return the initial counter configuration of the machine."""
        return (1,)

    @property
    def encoded_configuration_size(self) -> int:
        """Return the size of the encoded counter configuration."""
        return 2

    def sample_counter_configurations(self) -> list[tuple[int, ...]]:
        """Return a sample counter configuration."""
        return self._get_possible_counter_configurations()

    def _get_state_transition_function(self) -> dict:
        """Return the state transition function."""
        return {
            0: {
                "T_1 / (Z)": 1,
                "T_1 / (NZ)": 0,
                "DEFAULT / (-)": 0,
            },
            1: {
                "T_2 / (Z)": 2,
                "T_2 / (NZ)": 1,
                "DEFAULT / (-)": 1,
            },
            2: {
                "T_3 / (Z)": -1,
                "T_3 / (NZ)": 2,
                "DEFAULT / (-)": 2,
            },
        }

    def _get_counter_transition_function(self) -> dict:
        """Return the counter transition function."""
        return {
            0: {
                "T_1 / (Z)": (0,),
                "T_1 / (NZ)": (-1,),
                "DEFAULT / (-)": (0,),
            },
            1: {
                "T_2 / (Z)": (0,),
                "T_2 / (NZ)": (-1,),
                "DEFAULT / (-)": (0,),
            },
            2: {
                "T_3 / (Z)": (0,),
                "T_3 / (NZ)": (-1,),
                "DEFAULT / (-)": (0,),
            },
        }

    def _get_reward_transition_function(self) -> dict:
        """Return the reward transition function."""
        return {
            0: {
                "T_1 / (Z)": 10,
                "T_1 / (NZ)": 10,
                "DEFAULT / (-)": self._create_nav_t_1_reward(),
            },
            1: {
                "T_2 / (Z)": 10,
                "T_2 / (NZ)": 10,
                "DEFAULT / (-)": self._create_nav_t_2_reward(),
            },
            2: {
                "T_3 / (Z)": 10,
                "T_3 / (NZ)": 10,
                "DEFAULT / (-)": self._create_nav_t_3_reward(),
            },
        }

    def _get_possible_counter_configurations(self) -> list[tuple[int, ...]]:
        """Return the possible counter configurations."""
        return [
            (0,),
            (1,),
        ]

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
