import pytest

from pycrm.automaton import RewardMachine, RmToCrmAdapter


class TestC0Implementation:
    """Test the c_0 property implementation handling."""

    def test_reward_machine_with_c_0_works(self):
        """Test that reward machines with c_0 implemented work correctly."""
        # Import here to avoid fixture instantiation
        from tests.automaton.machine.emulation.conftest import RewardMachineWithC0

        rm = RewardMachineWithC0()
        assert rm.c_0 == (0,)
        assert rm.u_0 == 0

    def test_reward_machine_without_c_0_works(self, reward_machine_without_c0):
        """Test that reward machines without c_0 work correctly and get default c_0."""
        rm = reward_machine_without_c0

        # Should have c_0 property set to (0,)
        assert rm.c_0 == (0,)
        assert hasattr(rm, "c_0")
        assert rm.c_0 == (0,)

    def test_counting_reward_machine_with_c_0_works(
        self, counting_reward_machine_with_c0
    ):
        """Test that counting reward machines with c_0 implemented work correctly."""
        crm = counting_reward_machine_with_c0

        # Should use the implemented c_0
        assert crm.c_0 == (0,)

    def test_counting_reward_machine_without_c_0_raises_error(self):
        """Test that counting reward machines without c_0 raise an error."""
        with pytest.raises(
            TypeError,
            match=r"Can't instantiate abstract class.*"
            r"(?:with|without an implementation for) abstract method",
        ):
            # Import here to avoid fixture instantiation
            from tests.automaton.machine.emulation.conftest import (
                CountingRewardMachineWithoutC0,
            )

            CountingRewardMachineWithoutC0()

    def test_machine_type_detection(
        self, reward_machine_without_c0, counting_reward_machine_with_c0
    ):
        """Test that machine types can be correctly identified."""
        from pycrm.automaton import CountingRewardMachine

        # Test with reward machine (inherits from CountingRewardMachine
        # but acts like RM)
        rm = reward_machine_without_c0
        assert isinstance(rm, CountingRewardMachine)
        assert rm.c_0 == (0,)

        # Test with counting reward machine
        crm = counting_reward_machine_with_c0
        assert isinstance(crm, CountingRewardMachine)
        assert crm.c_0 == (0,)

    def test_c_0_property_accessibility(
        self, reward_machine_without_c0, counting_reward_machine_with_c0
    ):
        """Test that c_0 property is accessible after initialization."""
        # Reward machine
        rm = reward_machine_without_c0
        assert hasattr(rm, "c_0")
        assert callable(type(rm).c_0.fget)

        # Counting reward machine
        crm = counting_reward_machine_with_c0
        assert hasattr(crm, "c_0")
        assert callable(type(crm).c_0.fget)

    def test_c_0_is_tuple(
        self, reward_machine_without_c0, counting_reward_machine_with_c0
    ):
        """Test that c_0 always returns a tuple."""
        # Reward machine
        rm = reward_machine_without_c0
        assert isinstance(rm.c_0, tuple)

        # Counting reward machine
        crm = counting_reward_machine_with_c0
        assert isinstance(crm.c_0, tuple)

    def test_c_0_immutability(
        self, reward_machine_without_c0, counting_reward_machine_with_c0
    ):
        """Test that c_0 returns a consistent value."""
        # Reward machine
        rm = reward_machine_without_c0
        c_0_first = rm.c_0
        c_0_second = rm.c_0
        assert c_0_first == c_0_second
        assert c_0_first == (0,)

        # Counting reward machine
        crm = counting_reward_machine_with_c0
        c_0_first = crm.c_0
        c_0_second = crm.c_0
        assert c_0_first == c_0_second
        assert c_0_first == (0,)


class TestRmToCrmAdapter:
    """Test the RmToCrmAdapter class."""

    def test_adapter_initialization(self, rm: RewardMachine):
        """Test that the adapter can be initialized with a reward machine."""
        adapter = RmToCrmAdapter(rm)
        assert adapter._rm is rm
        assert adapter.env_prop_enum is rm.env_prop_enum

    def test_adapter_u_0_property(self, rm: RewardMachine):
        """Test that the adapter returns the correct initial state."""
        adapter = RmToCrmAdapter(rm)
        assert adapter.u_0 == rm.u_0

    def test_adapter_c_0_property(self, rm: RewardMachine):
        """Test that the adapter returns the correct initial counter configuration."""
        adapter = RmToCrmAdapter(rm)
        assert adapter.c_0 == (0,)

    def test_adapter_state_transition_function(self, rm: RewardMachine):
        """Test that the adapter correctly converts state transition functions."""
        adapter = RmToCrmAdapter(rm)

        # Get the converted transition function
        converted_delta_u = adapter._get_state_transition_function()

        # Check that each transition expression has been modified with " / (Z)"
        for state, transitions in rm._get_state_transition_function().items():
            assert state in converted_delta_u
            for expr in transitions.keys():
                expected_expr = f"{expr} / (Z)"
                assert expected_expr in converted_delta_u[state]
                assert converted_delta_u[state][expected_expr] == transitions[expr]

    def test_adapter_reward_transition_function(self, rm: RewardMachine):
        """Test that the adapter correctly converts reward transition functions."""
        adapter = RmToCrmAdapter(rm)

        # Get the converted transition function
        converted_delta_r = adapter._get_reward_transition_function()

        # Check that each transition expression has been modified with " / (Z)"
        for state, transitions in rm._get_reward_transition_function().items():
            assert state in converted_delta_r
            for expr in transitions.keys():
                expected_expr = f"{expr} / (Z)"
                assert expected_expr in converted_delta_r[state]
                assert converted_delta_r[state][expected_expr] == transitions[expr]

    def test_adapter_counter_transition_function(self, rm: RewardMachine):
        """Test that the adapter creates appropriate counter transition functions."""
        adapter = RmToCrmAdapter(rm)

        # Get the counter transition function
        delta_c = adapter._get_counter_transition_function()

        # Check that each transition expression has been created with (0,) counter delta
        for state, transitions in rm._get_state_transition_function().items():
            assert state in delta_c
            for expr in transitions.keys():
                expected_expr = f"{expr} / (Z)"
                assert expected_expr in delta_c[state]
                assert delta_c[state][expected_expr] == (0,)

    def test_adapter_sample_counter_configurations(self, rm: RewardMachine):
        """Test that the adapter returns the correct counter configurations."""
        adapter = RmToCrmAdapter(rm)
        configs = adapter.sample_counter_configurations()
        assert configs == [(0,)]

    def test_adapter_transition_functionality(self, rm: RewardMachine):
        """Test that the adapter can perform transitions correctly."""
        adapter = RmToCrmAdapter(rm)

        # Test a transition from initial state with no events
        u_next, c_next, reward_fn = adapter.transition(0, (0,), set())

        # The adapter should work like a CRM with a single counter always at 0
        assert u_next == rm.u_0  # Should stay in initial state
        assert c_next == (0,)  # Counter should remain 0
        assert callable(reward_fn)

    def test_adapter_encoding_methods(self, rm: RewardMachine):
        """Test that the adapter has the encoding methods from CountingRewardMachine."""
        adapter = RmToCrmAdapter(rm)

        # Test machine state encoding - adapter should have terminal state handling
        u_enc = adapter.encode_machine_state(0)
        assert len(u_enc) >= 2  # At least initial and terminal state

        # Test counter configuration encoding
        c_enc = adapter.encode_counter_configuration((0,))
        assert c_enc == [0.0]  # Single counter at 0

        # Test counter state encoding
        cs_enc = adapter.encode_counter_state((0,))
        assert cs_enc == [0]  # Counter at 0 is encoded as 0
