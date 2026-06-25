import pytest

from pycrm.automaton import CountingRewardMachine
from tests.conftest import EnvProps


class ValidatedCRM(CountingRewardMachine):
    """Counting reward machine used to test construction-time validation."""

    def __init__(self) -> None:
        """Initialise the counting reward machine."""
        super().__init__(env_prop_enum=EnvProps)

    @property
    def u_0(self) -> int:
        """Return the initial state."""
        return 0

    @property
    def c_0(self) -> tuple[int, ...]:
        """Return the initial counter configuration."""
        return (0,)

    def _get_state_transition_function(self) -> dict:
        """Return the state transition function."""
        return {0: {"EVENT_A / (Z)": 1}}

    def _get_counter_transition_function(self) -> dict:
        """Return the counter transition function."""
        return {0: {"EVENT_A / (Z)": (0,)}}

    def _get_reward_transition_function(self) -> dict:
        """Return the reward transition function."""
        return {0: {"EVENT_A / (Z)": 1}}

    def sample_counter_configurations(self) -> list[tuple[int, ...]]:
        """Return sampled counter configurations."""
        return [(0,)]


class CounterTestArityMismatchCRM(ValidatedCRM):
    """CRM with a transition expression that tests too many counters."""

    def _get_state_transition_function(self) -> dict:
        """Return the state transition function."""
        return {0: {"EVENT_A / (Z,NZ)": 1}}

    def _get_counter_transition_function(self) -> dict:
        """Return the counter transition function."""
        return {0: {"EVENT_A / (Z,NZ)": (0,)}}

    def _get_reward_transition_function(self) -> dict:
        """Return the reward transition function."""
        return {0: {"EVENT_A / (Z,NZ)": 1}}


class CounterDeltaArityMismatchCRM(ValidatedCRM):
    """CRM with a counter delta that has the wrong arity."""

    def _get_counter_transition_function(self) -> dict:
        """Return the counter transition function."""
        return {0: {"EVENT_A / (Z)": (0, 0)}}


class SampleCounterArityMismatchCRM(ValidatedCRM):
    """CRM with sampled counter configurations that have the wrong arity."""

    def sample_counter_configurations(self) -> list[tuple[int, ...]]:
        """Return sampled counter configurations."""
        return [(0, 0)]


class UnexpectedCounterTransitionCRM(ValidatedCRM):
    """CRM with an extra counter-transition key."""

    def _get_counter_transition_function(self) -> dict:
        """Return the counter transition function."""
        return {
            0: {
                "EVENT_A / (Z)": (0,),
                "EVENT_B / (Z)": (0,),
            }
        }


class InvalidInitialCounterConfigurationCRM(ValidatedCRM):
    """CRM with an invalid initial counter configuration."""

    @property
    def c_0(self) -> list[int]:  # type: ignore[override]
        """Return the initial counter configuration."""
        return [0]


def test_valid_machine_spec_initialises() -> None:
    """Test a valid machine spec passes construction-time validation."""
    crm = ValidatedCRM()

    assert crm.u_0 == 0
    assert crm.c_0 == (0,)


def test_transition_counter_test_arity_is_validated_during_initialisation() -> None:
    """Test transition counter-test arity is checked before runtime."""
    with pytest.raises(ValueError) as exc_info:
        CounterTestArityMismatchCRM()

    assert "expects 2 counter value(s), but c_0 has 1" in str(exc_info.value)


def test_counter_delta_arity_is_validated_during_initialisation() -> None:
    """Test counter delta arity is checked before runtime."""
    with pytest.raises(ValueError) as exc_info:
        CounterDeltaArityMismatchCRM()

    assert "Counter delta for transition 0: 'EVENT_A / (Z)'" in str(exc_info.value)
    assert "has arity 2, but c_0 has 1" in str(exc_info.value)


def test_sample_counter_configurations_are_validated_during_initialisation() -> None:
    """Test sampled counter configurations are checked before runtime."""
    with pytest.raises(ValueError) as exc_info:
        SampleCounterArityMismatchCRM()

    assert "Sample counter configuration 0 has arity 2" in str(exc_info.value)


def test_unexpected_counter_transition_key_is_validated_during_initialisation() -> None:
    """Test extra transition-map keys are rejected before runtime."""
    with pytest.raises(ValueError) as exc_info:
        UnexpectedCounterTransitionCRM()

    assert "Unexpected counter transition for state 0: EVENT_B / (Z)" in str(
        exc_info.value
    )


def test_initial_counter_configuration_must_be_tuple() -> None:
    """Test c_0 must be a tuple."""
    with pytest.raises(ValueError) as exc_info:
        InvalidInitialCounterConfigurationCRM()

    assert "Initial counter configuration c_0 must be a tuple" in str(exc_info.value)
