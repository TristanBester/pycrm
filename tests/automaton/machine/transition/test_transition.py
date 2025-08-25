from typing import Any

import pytest

from pycrm.automaton import CountingRewardMachine, RmToCrmAdapter
from tests.conftest import EnvProps


def transition_cases() -> list[dict[str, Any]]:
    """Return test cases for the transition function."""
    return [
        {
            "description": "Transition from state 0 to state 0",
            "u": 0,
            "c": (0,),
            "props": [],
            "u_next": 0,
            "c_next": (0,),
            "reward": 0,
        },
        {
            "description": "Transition from state 0 to state 1 with EVENT_A",
            "u": 0,
            "c": (0,),
            "props": [EnvProps.EVENT_A],
            "u_next": 0,
            "c_next": (1,),
            "reward": 0,
        },
        {
            "description": "Transition from state 0 to state 2 with EVENT_A",
            "u": 0,
            "c": (1,),
            "props": [EnvProps.EVENT_A],
            "u_next": 0,
            "c_next": (2,),
            "reward": 0,
        },
        {
            "description": "Transition from state 0 to state 1 with EVENT_B",
            "u": 0,
            "c": (2,),
            "props": [EnvProps.EVENT_B],
            "u_next": 1,
            "c_next": (1,),
            "reward": 0,
        },
        {
            "description": "Transition from state 1 to state 0 with EVENT_A",
            "u": 1,
            "c": (0,),
            "props": [EnvProps.EVENT_A],
            "u_next": 1,
            "c_next": (0,),
            "reward": 0,
        },
        {
            "description": "Transition from state 1 to state 1 with EVENT_A",
            "u": 1,
            "c": (1,),
            "props": [EnvProps.EVENT_A],
            "u_next": 1,
            "c_next": (1,),
            "reward": 0,
        },
        {
            "description": "Transition from state 1 to state 1 with EVENT_B",
            "u": 1,
            "c": (2,),
            "props": [EnvProps.EVENT_B],
            "u_next": 1,
            "c_next": (1,),
            "reward": 0,
        },
        {
            "description": "Transition from state 1 to state 0 with EVENT_B",
            "u": 1,
            "c": (1,),
            "props": [EnvProps.EVENT_B],
            "u_next": 1,
            "c_next": (0,),
            "reward": 0,
        },
        {
            "description": "Transition from state 1 to state 2 with EVENT_B",
            "u": 1,
            "c": (0,),
            "props": [EnvProps.EVENT_B],
            "u_next": 2,
            "c_next": (0,),
            "reward": 1,
        },
    ]


class TestTransition:
    """Test the transition function."""

    @pytest.mark.parametrize(
        "test_case",
        transition_cases(),
        ids=lambda test_case: test_case["description"],
    )
    def test_defined_state_transition_success(
        self, test_case: dict[str, Any], crm: CountingRewardMachine
    ) -> None:
        """Test that a defined state transition succeeds."""
        u_next_actual, c_next_actual, reward_fn_actual = crm.transition(
            u=test_case["u"],
            c=test_case["c"],
            props=test_case["props"],
        )
        assert u_next_actual == test_case["u_next"]
        assert c_next_actual == test_case["c_next"]
        assert reward_fn_actual(None, None, None) == test_case["reward"]

    def test_terminal_state_transition_failure(
        self, crm: CountingRewardMachine
    ) -> None:
        """Test that a transition from a terminal state fails."""
        with pytest.raises(ValueError) as exc_info:
            crm.transition(2, (0,), set())
        assert "State u=2 is terminal or not defined in the transition function" in str(
            exc_info.value
        )

    def test_undefined_state_transition_failure(
        self, crm: CountingRewardMachine
    ) -> None:
        """Test that a transition to an undefined state fails."""
        with pytest.raises(ValueError) as exc_info:
            crm.transition(0, (0,), {EnvProps.EVENT_B})
        assert "Transition not defined for machine configuration" in str(exc_info.value)


class TestAdapterTransition:
    """Test transition functionality of the RmToCrmAdapter."""

    def test_adapter_transition_with_no_events(
        self, rm_to_crm_adapter: RmToCrmAdapter
    ) -> None:
        """Test that the adapter handles transitions with no events correctly."""
        adapter = rm_to_crm_adapter

        # Test transition from initial state with no events
        u_next, c_next, reward_fn = adapter.transition(0, (0,), set())

        # Should stay in initial state with counter at 0
        assert u_next == 0
        assert c_next == (0,)
        assert callable(reward_fn)

    def test_adapter_transition_with_events(
        self, rm_to_crm_adapter: RmToCrmAdapter
    ) -> None:
        """Test that the adapter handles transitions with events correctly."""
        adapter = rm_to_crm_adapter

        # Test transition with EVENT_A
        _, _c_next, _reward_fn = adapter.transition(0, (0,), {EnvProps.EVENT_A})

        # Should transition based on the underlying RM logic
        assert callable(_reward_fn)

    def test_adapter_terminal_state_handling(
        self, rm_to_crm_adapter: RmToCrmAdapter
    ) -> None:
        """Test that the adapter handles terminal states correctly."""
        adapter = rm_to_crm_adapter

        # Transition to state 1, then to terminal state
        u_next, _c_next, _reward_fn = adapter.transition(0, (0,), {EnvProps.EVENT_B})
        assert u_next == 1  # Should go to state 1

        # Now transition to terminal state
        u_final, c_final, reward_fn_final = adapter.transition(
            1,
            (0,),
            {EnvProps.EVENT_B},
        )
        assert u_final == adapter.F[0]  # Should be terminal state
        assert c_final == (0,)  # Counter should remain 0
        assert callable(reward_fn_final)

    def test_adapter_terminal_state_error(
        self, rm_to_crm_adapter: RmToCrmAdapter
    ) -> None:
        """Test that transitioning from terminal state raises error."""
        adapter = rm_to_crm_adapter

        # First get to terminal state
        u_terminal = adapter.F[0]

        # Try to transition from terminal state - should raise error
        with pytest.raises(ValueError) as exc_info:
            adapter.transition(u_terminal, (0,), set())
        assert "State u=" in str(exc_info.value)
        assert "is terminal or not defined" in str(exc_info.value)
