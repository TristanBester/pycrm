import pytest

from pycrm.automaton import CountingRewardMachine, RmToCrmAdapter
from tests.conftest import EnvProps


class TestTerminalStateReplacement:
    """Test the replacement of the terminal state flag in the CountingRewardMachine."""

    def test_terminal_state_flag_replaced(self, ccrm: CountingRewardMachine) -> None:
        """Test terminal state flag is replaced with the terminal state index."""
        destination_states = []
        for _, next_states in ccrm._delta_u.items():
            for _, next_state in next_states.items():
                destination_states.append(next_state)
        assert -1 not in destination_states

    def test_non_terminal_states(self, ccrm: CountingRewardMachine) -> None:
        """Test that the non-terminal states are correctly identified."""
        assert ccrm.U == [0, 1]

    def test_terminal_states(self, ccrm: CountingRewardMachine) -> None:
        """Test that the terminal states are correctly identified."""
        assert ccrm.F == [2]

    def test_terminal_state_flag_replacement_does_not_affect_reward_function_crm(
        self, crm: CountingRewardMachine
    ) -> None:
        """Test terminal state flag replacement does not affect the reward function."""
        reward_fn = crm._delta_r[1]["EVENT_B / (Z)"]
        assert reward_fn(None, None, None) == 1

    def test_terminal_state_flag_replacement_does_not_affect_counter_transition_crm(
        self, crm: CountingRewardMachine
    ) -> None:
        """Test flag replacement does not affect counter transition function."""
        assert crm._delta_c[1]["EVENT_B / (Z)"] == (0,)

    def test_terminal_state_flag_replacement_does_not_affect_reward_function_ccrm(
        self, ccrm: CountingRewardMachine
    ) -> None:
        """Test terminal state flag replacement does not affect the reward function."""
        reward_fn = ccrm._delta_r[1]["EVENT_B / (Z)"]
        assert reward_fn(None, None, None) == 1

    def test_terminal_state_flag_replacement_does_not_affect_counter_transition_ccrm(
        self, ccrm: CountingRewardMachine
    ) -> None:
        """Test flag replacement does not affect counter transition function."""
        assert ccrm._delta_c[1]["EVENT_B / (Z)"] == (0,)


class TestAdapterTerminalStateHandling:
    """Test terminal state handling in the RmToCrmAdapter."""

    def test_adapter_terminal_state_replacement(
        self, rm_to_crm_adapter: RmToCrmAdapter
    ):
        """Test that the adapter correctly replaces terminal state flags."""
        adapter = rm_to_crm_adapter

        # Check that no -1 flags remain in the transition functions
        destination_states = []
        for _, next_states in adapter._delta_u.items():
            for _, next_state in next_states.items():
                destination_states.append(next_state)
        assert -1 not in destination_states

    def test_adapter_terminal_states_identification(
        self, rm_to_crm_adapter: RmToCrmAdapter
    ):
        """Test that the adapter correctly identifies terminal states."""
        adapter = rm_to_crm_adapter

        # The adapter should have terminal states
        assert len(adapter.F) > 0
        assert all(
            state >= 0 for state in adapter.F
        )  # Terminal states should be positive

    def test_adapter_non_terminal_states(self, rm_to_crm_adapter: RmToCrmAdapter):
        """Test that the adapter correctly identifies non-terminal states."""
        adapter = rm_to_crm_adapter

        # Should have the states from the underlying RM plus terminal state
        expected_states = [0, 1]  # From our RM fixture
        assert all(state in adapter.U for state in expected_states)

    def test_adapter_terminal_state_transitions(
        self, rm_to_crm_adapter: RmToCrmAdapter
    ):
        """Test that the adapter correctly handles terminal state transitions."""
        adapter = rm_to_crm_adapter

        # Transition to terminal state
        u_next, c_next, reward_fn = adapter.transition(1, (0,), {EnvProps.EVENT_B})

        # Should be terminal state
        assert u_next in adapter.F
        assert c_next == (0,)  # Counter should remain 0

        # Reward function should be callable
        assert callable(reward_fn)

    def test_adapter_terminal_state_error_handling(
        self, rm_to_crm_adapter: RmToCrmAdapter
    ):
        """Test adapter handles errors when transitioning from terminal states."""
        adapter = rm_to_crm_adapter

        # Get terminal state
        terminal_state = adapter.F[0]

        # Trying to transition from terminal state should raise error
        with pytest.raises(ValueError) as exc_info:
            adapter.transition(terminal_state, (0,), set())
        assert "State u=" in str(exc_info.value)
