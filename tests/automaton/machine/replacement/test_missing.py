import pytest

from crm.automaton import CountingRewardMachine
from tests.conftest import EnvProps


class CRM(CountingRewardMachine):
    """Concrete implementation of a counting reward machine."""

    def __init__(self) -> None:
        """Initialise the counting reward machine."""
        super().__init__(env_prop_enum=EnvProps)

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


class MissingCounterTransitionCRM(CRM):
    """Concrete implementation of a counting reward machine."""

    def _get_counter_transition_function(self) -> dict:
        """Return the counter transition function."""
        return {
            0: {
                "not EVENT_A and not EVENT_B / (-)": (0,),
                "EVENT_B / (NZ)": (-1,),
            },
            1: {
                "not EVENT_B / (-)": (0,),
                "EVENT_B / (NZ)": (-1,),
                "EVENT_B / (Z)": (0,),
            },
        }


class MissingRewardTransitionCRM(CRM):
    """Concrete implementation of a counting reward machine."""

    def _get_reward_transition_function(self) -> dict:
        """Return the reward transition function."""
        return {
            0: {
                "not EVENT_A and not EVENT_B / (-)": (
                    self._create_constant_reward_function(0)
                ),
                "EVENT_B / (NZ)": self._create_constant_reward_function(0),
            },
            1: {
                "not EVENT_B / (-)": self._create_constant_reward_function(0),
                "EVENT_B / (NZ)": self._create_constant_reward_function(0),
                "EVENT_B / (Z)": self._create_constant_reward_function(1),
            },
        }


class TestMissingTransitions:
    """Test the missing transitions."""

    def test_missing_counter_transition_raises(self) -> None:
        """Test that a missing counter transition raises an error."""
        with pytest.raises(ValueError) as exc_info:
            MissingCounterTransitionCRM()
        assert "Missing counter configuration for transition" in str(exc_info.value)

    def test_missing_reward_transition_raises(self) -> None:
        """Test that a missing reward transition raises an error."""
        with pytest.raises(ValueError) as exc_info:
            MissingRewardTransitionCRM()
        assert "Missing reward function for transition" in str(exc_info.value)
