import pytest


class TestC0Implementation:
    """Test the c_0 property implementation handling."""

    def test_reward_machine_with_c_0_raises_error(self):
        """Test that reward machines with c_0 implemented raise an error."""
        with pytest.raises(
            ValueError, match=r"Reward machines do not have a c_0 property"
        ):
            # Import here to avoid fixture instantiation
            from tests.automaton.machine.emulation.conftest import RewardMachineWithC0

            RewardMachineWithC0()

    def test_reward_machine_without_c_0_works(self, reward_machine_without_c0):
        """Test that reward machines without c_0 work correctly and get default c_0."""
        rm = reward_machine_without_c0

        # Should have default c_0 set to (0,)
        assert rm.c_0 == (0,)
        assert hasattr(rm, "_c_0")
        assert rm._c_0 == (0,)

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
            ValueError, match=r"Counting reward machines must have a c_0 property"
        ):
            # Import here to avoid fixture instantiation
            from tests.automaton.machine.emulation.conftest import (
                CountingRewardMachineWithoutC0,
            )

            CountingRewardMachineWithoutC0()

    def test_is_reward_machine_detection(
        self, reward_machine_without_c0, counting_reward_machine_with_c0
    ):
        """Test that _is_reward_machine correctly identifies machine types."""
        # Test with reward machine (no counter conditions in expressions)
        rm = reward_machine_without_c0
        assert rm._is_reward_machine() is True

        # Test with counting reward machine (has counter conditions with " / ")
        crm = counting_reward_machine_with_c0
        assert crm._is_reward_machine() is False

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
        # Reward machine (uses default implementation, so has _c_0)
        rm = reward_machine_without_c0
        c_0_first = rm.c_0
        c_0_second = rm.c_0
        assert c_0_first == c_0_second
        assert c_0_first is rm._c_0

        # Counting reward machine (has custom implementation, no _c_0)
        crm = counting_reward_machine_with_c0
        c_0_first = crm.c_0
        c_0_second = crm.c_0
        assert c_0_first == c_0_second
        # Custom implementations don't use _c_0, so just check consistency
        assert c_0_first == (0,)
