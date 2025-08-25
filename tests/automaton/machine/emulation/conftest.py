import pytest

from pycrm.automaton import CountingRewardMachine
from tests.conftest import EnvProps


class RewardMachineWithC0(CountingRewardMachine):
    """A reward machine that incorrectly implements c_0 property."""

    def __init__(self) -> None:
        """Initialise the reward machine."""
        super().__init__(env_prop_enum=EnvProps)

    @property
    def u_0(self) -> int:
        """Return the initial state of the machine."""
        return 0

    @property
    def c_0(self) -> tuple[int, ...]:
        """Return the initial counter configuration (should not be implemented)."""
        return (0,)

    @property
    def encoded_configuration_size(self) -> int:
        """Return the size of the encoded configuration."""
        return 1

    def _get_state_transition_function(self) -> dict:
        """Return the state transition function for reward machine (no counters)."""
        return {
            0: {
                "not EVENT_A": 0,
                "EVENT_A": 1,
            },
            1: {
                "EVENT_B": -1,
                "not EVENT_B": 1,
            },
        }

    def _get_counter_transition_function(self) -> dict:
        """Return empty counter transition function for reward machine."""
        return {
            0: {
                "not EVENT_A": (0,),
                "EVENT_A": (0,),
            },
            1: {
                "EVENT_B": (0,),
                "not EVENT_B": (0,),
            },
        }

    def _get_reward_transition_function(self) -> dict:
        """Return the reward transition function."""
        return {
            0: {
                "not EVENT_A": 0,
                "EVENT_A": 0,
            },
            1: {
                "EVENT_B": 1,
                "not EVENT_B": 0,
            },
        }

    def sample_counter_configurations(self) -> list[tuple[int, ...]]:
        """Return sample counter configurations."""
        return [(0,)]


class RewardMachineWithoutC0(CountingRewardMachine):
    """A reward machine that correctly does not implement c_0 property."""

    def __init__(self) -> None:
        """Initialise the reward machine."""
        super().__init__(env_prop_enum=EnvProps)

    @property
    def u_0(self) -> int:
        """Return the initial state of the machine."""
        return 0

    @property
    def c_0(self) -> tuple[int, ...]:
        """Return the initial counter configuration."""
        return (0,)

    @property
    def encoded_configuration_size(self) -> int:
        """Return the size of the encoded configuration."""
        return 1

    def _get_state_transition_function(self) -> dict:
        """Return the state transition function for reward machine (no counters)."""
        return {
            0: {
                "not EVENT_A": 0,
                "EVENT_A": 1,
            },
            1: {
                "EVENT_B": -1,
                "not EVENT_B": 1,
            },
        }

    def _get_counter_transition_function(self) -> dict:
        """Return empty counter transition function for reward machine."""
        return {
            0: {
                "not EVENT_A": (0,),
                "EVENT_A": (0,),
            },
            1: {
                "EVENT_B": (0,),
                "not EVENT_B": (0,),
            },
        }

    def _get_reward_transition_function(self) -> dict:
        """Return the reward transition function."""
        return {
            0: {
                "not EVENT_A": 0,
                "EVENT_A": 0,
            },
            1: {
                "EVENT_B": 1,
                "not EVENT_B": 0,
            },
        }

    def sample_counter_configurations(self) -> list[tuple[int, ...]]:
        """Return sample counter configurations."""
        return [(0,)]


class CountingRewardMachineWithC0(CountingRewardMachine):
    """A counting reward machine that correctly implements c_0 property."""

    def __init__(self) -> None:
        """Initialise the counting reward machine."""
        super().__init__(env_prop_enum=EnvProps)

    @property
    def u_0(self) -> int:
        """Return the initial state of the machine."""
        return 0

    @property
    def c_0(self) -> tuple[int, ...]:
        """Return the initial counter configuration."""
        return (0,)

    @property
    def encoded_configuration_size(self) -> int:
        """Return the size of the encoded configuration."""
        return 2

    def _get_state_transition_function(self) -> dict:
        """Return the state transition function with counter conditions."""
        return {
            0: {
                "EVENT_A / (-)": 0,
                "EVENT_B / (NZ)": 1,
            },
            1: {
                "EVENT_B / (Z)": -1,
                "not EVENT_B / (-)": 1,
            },
        }

    def _get_counter_transition_function(self) -> dict:
        """Return counter transition function."""
        return {
            0: {
                "EVENT_A / (-)": (1,),
                "EVENT_B / (NZ)": (-1,),
            },
            1: {
                "EVENT_B / (Z)": (0,),
                "not EVENT_B / (-)": (0,),
            },
        }

    def _get_reward_transition_function(self) -> dict:
        """Return the reward transition function."""
        return {
            0: {
                "EVENT_A / (-)": 0,
                "EVENT_B / (NZ)": 0,
            },
            1: {
                "EVENT_B / (Z)": 1,
                "not EVENT_B / (-)": 0,
            },
        }

    def sample_counter_configurations(self) -> list[tuple[int, ...]]:
        """Return sample counter configurations."""
        return [(0,), (1,), (2,)]


class CountingRewardMachineWithoutC0(CountingRewardMachine):
    """A counting reward machine that incorrectly does not implement c_0 property."""

    def __init__(self) -> None:
        """Initialise the counting reward machine."""
        super().__init__(env_prop_enum=EnvProps)

    @property
    def u_0(self) -> int:
        """Return the initial state of the machine."""
        return 0

    @property
    def encoded_configuration_size(self) -> int:
        """Return the size of the encoded configuration."""
        return 2

    def _get_state_transition_function(self) -> dict:
        """Return the state transition function with counter conditions."""
        return {
            0: {
                "EVENT_A / (-)": 0,
                "EVENT_B / (NZ)": 1,
            },
            1: {
                "EVENT_B / (Z)": -1,
                "not EVENT_B / (-)": 1,
            },
        }

    def _get_counter_transition_function(self) -> dict:
        """Return counter transition function."""
        return {
            0: {
                "EVENT_A / (-)": (1,),
                "EVENT_B / (NZ)": (-1,),
            },
            1: {
                "EVENT_B / (Z)": (0,),
                "not EVENT_B / (-)": (0,),
            },
        }

    def _get_reward_transition_function(self) -> dict:
        """Return the reward transition function."""
        return {
            0: {
                "EVENT_A / (-)": 0,
                "EVENT_B / (NZ)": 0,
            },
            1: {
                "EVENT_B / (Z)": 1,
                "not EVENT_B / (-)": 0,
            },
        }

    def sample_counter_configurations(self) -> list[tuple[int, ...]]:
        """Return sample counter configurations."""
        return [(0,), (1,), (2,)]


@pytest.fixture
def reward_machine_with_c0() -> RewardMachineWithC0:
    """Fixture for a reward machine that incorrectly implements c_0 property."""
    return RewardMachineWithC0()


@pytest.fixture
def reward_machine_without_c0() -> RewardMachineWithoutC0:
    """Fixture for a reward machine that correctly does not implement c_0 property."""
    return RewardMachineWithoutC0()


@pytest.fixture
def counting_reward_machine_with_c0() -> CountingRewardMachineWithC0:
    """Fixture for a counting reward machine that correctly implements c_0 property."""
    return CountingRewardMachineWithC0()


@pytest.fixture
def counting_reward_machine_without_c0() -> CountingRewardMachineWithoutC0:
    """Fixture for CRM incorrectly does not implement c_0 property."""
    return CountingRewardMachineWithoutC0()
