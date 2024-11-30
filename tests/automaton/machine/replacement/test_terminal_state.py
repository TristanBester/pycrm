from crm.automaton import CountingRewardMachine


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
