from enum import Enum, auto

import gymnasium as gym
import numpy as np
import pytest

from pycrm.automaton import CountingRewardMachine, RewardMachine, RmToCrmAdapter
from pycrm.crossproduct import CrossProduct
from pycrm.label import LabellingFunction


class Events(Enum):
    """Events."""

    EVENT_A = auto()
    EVENT_B = auto()


class GroundEnv(gym.Env):
    """Ground environment."""

    def __init__(self):
        """Initialize the ground environment."""
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,))
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,))

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset the ground environment."""
        super().reset(seed=seed, options=options)
        return np.array([0]), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Step the ground environment."""
        return np.array([1]), 0.0, False, False, {}


class LabelFunction(LabellingFunction[np.ndarray, np.ndarray]):
    """Label function."""

    @LabellingFunction.event
    def test_event_a(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> Enum | None:
        """Test for presence of EVENT_A."""
        if obs[0] == 0:
            return Events.EVENT_A
        return None

    @LabellingFunction.event
    def test_event_b(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> Enum | None:
        """Test for presence of EVENT_B."""
        if obs[0] == 1:
            return Events.EVENT_B
        return None


class CRM(CountingRewardMachine):
    """Counting reward machine."""

    @property
    def u_0(self) -> int:
        """Return the initial state of the machine."""
        return 0

    @property
    def c_0(self) -> tuple[int]:
        """Return the initial counter configuration of the machine."""
        return (0,)

    @property
    def encoded_configuration_size(self) -> int:
        """Return the size of the encoded configuration."""
        return 2

    def _get_state_transition_function(self) -> dict:
        """Return the state transition function."""
        return {
            0: {
                "EVENT_A / (-)": 1,
                "EVENT_B / (-)": 0,
            },
            1: {
                "EVENT_A / (-)": 2,
                "EVENT_B / (-)": 0,
            },
        }

    def _get_counter_transition_function(self) -> dict:
        """Return the counter transition function."""
        return {
            0: {
                "EVENT_A / (-)": (1,),
                "EVENT_B / (-)": (0,),
            },
            1: {
                "EVENT_A / (-)": (0,),
                "EVENT_B / (-)": (1,),
            },
        }

    def _get_reward_transition_function(self) -> dict:
        """Return the reward transition function."""
        return {
            0: {
                "EVENT_A / (-)": 1,
                "EVENT_B / (-)": 0,
            },
            1: {
                "EVENT_A / (-)": 0,
                "EVENT_B / (-)": 1,
            },
        }

    def _get_possible_counter_configurations(self) -> list[tuple[int]]:
        """Return the possible counter configurations."""
        return [(0,), (1,), (2,)]

    def sample_counter_configurations(self) -> list[tuple[int]]:
        """Return a sample counter configuration."""
        return self._get_possible_counter_configurations()


class RM(RewardMachine):
    """Reward machine."""

    @property
    def u_0(self) -> int:
        """Return the initial state of the machine."""
        return 0

    def _get_state_transition_function(self) -> dict:
        """Return the state transition function."""
        return {
            0: {
                "EVENT_A": 1,
                "EVENT_B": 0,
            },
            1: {
                "EVENT_A": 2,
                "EVENT_B": 0,
            },
        }

    def _get_reward_transition_function(self) -> dict:
        """Return the reward transition function."""
        return {
            0: {
                "EVENT_A": 1,
                "EVENT_B": 0,
            },
            1: {
                "EVENT_A": 0,
                "EVENT_B": 1,
            },
        }


class CrossProductMDP(CrossProduct):
    """Cross product MDP."""

    def _get_obs(
        self, ground_obs: np.ndarray, u: int, c: tuple[int, ...]
    ) -> np.ndarray:
        """Get the cross product observation."""
        return np.array([ground_obs[0], u, c[0]])

    def to_ground_obs(self, obs: np.ndarray) -> np.ndarray:
        """Convert the cross product observation to the ground observation."""
        return obs[:2]


@pytest.fixture
def cross_product_mdp() -> CrossProductMDP:
    """Return a cross product MDP."""
    ground_env = GroundEnv()
    labelling_function = LabelFunction()
    crm = CRM(env_prop_enum=Events)
    return CrossProductMDP(
        ground_env=ground_env,
        machine=crm,
        lf=labelling_function,
        max_steps=10,
    )


@pytest.fixture
def rm() -> RM:
    """Return a reward machine."""
    return RM(env_prop_enum=Events)


@pytest.fixture
def rm_to_crm_adapter(rm: RM) -> RmToCrmAdapter:
    """Return a reward machine to counting reward machine adapter."""
    return RmToCrmAdapter(rm)


@pytest.fixture
def cross_product_mdp_with_adapter(
    rm_to_crm_adapter: RmToCrmAdapter,
) -> CrossProductMDP:
    """Return a cross product MDP using the RM to CRM adapter."""
    ground_env = GroundEnv()
    labelling_function = LabelFunction()
    return CrossProductMDP(
        ground_env=ground_env,
        machine=rm_to_crm_adapter,
        lf=labelling_function,
        max_steps=10,
    )
