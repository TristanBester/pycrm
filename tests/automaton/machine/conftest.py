import pytest

from crm.automaton import CountingRewardMachine
from tests.conftest import EnvProps


class CRM(CountingRewardMachine):
    """Concrete implementation of a counting reward machine."""

    def __init__(self) -> None:
        """Initialise the counting reward machine."""
        super().__init__(env_prop_enum=EnvProps)

    @property
    def u_0(self) -> int:
        """Return the initial state of the machine."""
        return 0

    @property
    def c_0(self) -> tuple[int, ...]:
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
                "not EVENT_A and not EVENT_B / (-)": 0,
                "EVENT_A / (-)": 0,
                "EVENT_B / (NZ)": 1,
            },
            1: {
                "not EVENT_B / (-)": 1,
                "EVENT_B / (NZ)": 1,
                "EVENT_B / (Z)": -1,
            },
        }

    def _get_counter_transition_function(self) -> dict:
        """Return the counter transition function."""
        return {
            0: {
                "not EVENT_A and not EVENT_B / (-)": (0,),
                "EVENT_A / (-)": (1,),
                "EVENT_B / (NZ)": (-1,),
            },
            1: {
                "not EVENT_B / (-)": (0,),
                "EVENT_B / (NZ)": (-1,),
                "EVENT_B / (Z)": (0,),
            },
        }

    def _get_reward_transition_function(self) -> dict:
        """Return the reward transition function."""
        return {
            0: {
                "not EVENT_A and not EVENT_B / (-)": (
                    self._create_constant_reward_function(0)
                ),
                "EVENT_A / (-)": self._create_constant_reward_function(0),
                "EVENT_B / (NZ)": self._create_constant_reward_function(0),
            },
            1: {
                "not EVENT_B / (-)": self._create_constant_reward_function(0),
                "EVENT_B / (NZ)": self._create_constant_reward_function(0),
                "EVENT_B / (Z)": self._create_constant_reward_function(1),
            },
        }

    def _get_possible_counter_configurations(self) -> list[tuple[int]]:
        """Return the possible counter configurations."""
        return [(0,), (1,)]

    def sample_counter_configurations(self) -> list[tuple[int]]:
        """Return a sample counter configuration."""
        return [(0,)]


class CCRM(CRM):
    """Concrete implementation of a constant counting reward machine."""

    def _get_reward_transition_function(self) -> dict:
        """Return the reward transition function."""
        return {
            0: {
                "not EVENT_A and not EVENT_B / (-)": 0,
                "EVENT_A / (-)": 0,
                "EVENT_B / (NZ)": 0,
            },
            1: {
                "not EVENT_B / (-)": 0,
                "EVENT_B / (NZ)": 0,
                "EVENT_B / (Z)": 1,
            },
        }


@pytest.fixture
def crm() -> CRM:
    """Fixture for a counting reward machine (CRM) for testing purposes."""
    return CRM()


@pytest.fixture
def ccrm() -> CCRM:
    """Fixture for a constant counting reward machine (CCRM) for testing purposes."""
    return CCRM()
