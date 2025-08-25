import pytest

from pycrm.automaton import CountingRewardMachine, RewardMachine, RmToCrmAdapter
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


class IncompleteRewardMachine(RewardMachine):
    """An incomplete reward machine for testing error handling."""

    @property
    def u_0(self) -> int:
        """Return the initial state of the machine."""
        return 0

    def _get_state_transition_function(self) -> dict:
        """Return the state transition function."""
        return {
            0: {
                "EVENT_A": 1,
                # Missing transition for "not EVENT_A"
            },
        }

    def _get_reward_transition_function(self) -> dict:
        """Return the reward transition function."""
        return {
            0: {
                "EVENT_A": 0,
                # Missing reward for "not EVENT_A"
            },
        }


class TestAdapterMissingTransitions:
    """Test missing transition handling in the RmToCrmAdapter."""

    def test_adapter_handles_missing_transitions_in_rm(self):
        """Test that the adapter can be created from an RM with missing transitions."""
        # This should work because the adapter converts RM transitions to CRM format
        rm = IncompleteRewardMachine(env_prop_enum=EnvProps)
        adapter = RmToCrmAdapter(rm)

        # The adapter should have been created successfully
        assert adapter._rm is rm
        assert adapter.env_prop_enum is rm.env_prop_enum

    def test_adapter_transition_with_missing_rm_transitions(self):
        """Test adapter handles transitions correctly even with incomplete RM."""
        rm = IncompleteRewardMachine(env_prop_enum=EnvProps)
        adapter = RmToCrmAdapter(rm)

        # This should work because the adapter adds the " / (Z)" suffix
        # and the underlying RM transition logic should handle it
        _, c_next, reward_fn = adapter.transition(0, (0,), {EnvProps.EVENT_A})

        # Should transition based on the RM logic
        assert c_next == (0,)  # Counter should always be 0 for RM adapter
        assert callable(reward_fn)

    def test_adapter_transition_with_undefined_rm_behavior(self):
        """Test that the adapter handles undefined transitions from the RM."""
        rm = IncompleteRewardMachine(env_prop_enum=EnvProps)
        adapter = RmToCrmAdapter(rm)

        # Test with EVENT_A which should work since the RM defines it
        u_next, c_next, reward_fn = adapter.transition(0, (0,), {EnvProps.EVENT_A})

        # Should handle the transition gracefully
        assert c_next == (0,)
        assert callable(reward_fn)
        assert u_next == 1  # From the RM's transition definition
