import pytest

from pycrm.automaton import RewardMachine, RmToCrmAdapter
from tests.conftest import EnvProps


class ConcreteRewardMachine(RewardMachine):
    """Concrete implementation of a reward machine for testing."""

    def __init__(self) -> None:
        """Initialise the reward machine."""
        super().__init__(env_prop_enum=EnvProps)

    @property
    def u_0(self) -> int:
        """Return the initial state of the machine."""
        return 0

    def _get_state_transition_function(self) -> dict:
        """Return the state transition function."""
        return {
            0: {
                "not EVENT_A and not EVENT_B": 0,
                "EVENT_A": 1,
                "EVENT_B": 0,
            },
            1: {
                "not EVENT_B": 1,
                "EVENT_B": -1,
            },
        }

    def _get_reward_transition_function(self) -> dict:
        """Return the reward transition function."""
        return {
            0: {
                "not EVENT_A and not EVENT_B": lambda obs, action, next_obs: 0,
                "EVENT_A": lambda obs, action, next_obs: 1,
                "EVENT_B": lambda obs, action, next_obs: 0,
            },
            1: {
                "not EVENT_B": lambda obs, action, next_obs: 0,
                "EVENT_B": lambda obs, action, next_obs: 10,
            },
        }


class TestRewardMachine:
    """Test the RewardMachine base class."""

    def test_reward_machine_initialization(self):
        """Test that a reward machine can be initialized."""
        rm = ConcreteRewardMachine()
        assert rm.env_prop_enum is EnvProps
        assert rm.u_0 == 0

    def test_reward_machine_abstract_methods(self):
        """Test that RewardMachine has the expected abstract methods."""
        rm = ConcreteRewardMachine()

        # Should have u_0 property
        assert hasattr(rm, "u_0")
        assert rm.u_0 == 0

        # Should have abstract methods implemented
        assert hasattr(rm, "_get_state_transition_function")
        assert hasattr(rm, "_get_reward_transition_function")

        # Should be able to call the methods
        state_fn = rm._get_state_transition_function()
        reward_fn = rm._get_reward_transition_function()

        assert isinstance(state_fn, dict)
        assert isinstance(reward_fn, dict)

    def test_reward_machine_state_transitions(self):
        """Test that reward machine state transitions work correctly."""
        rm = ConcreteRewardMachine()
        state_fn = rm._get_state_transition_function()

        # Check that all states have transition mappings
        assert 0 in state_fn
        assert 1 in state_fn

        # Check specific transitions
        assert "not EVENT_A and not EVENT_B" in state_fn[0]
        assert "EVENT_A" in state_fn[0]
        assert "EVENT_B" in state_fn[0]

        assert "not EVENT_B" in state_fn[1]
        assert "EVENT_B" in state_fn[1]

    def test_reward_machine_reward_functions(self):
        """Test that reward machine reward functions work correctly."""
        rm = ConcreteRewardMachine()
        reward_fn = rm._get_reward_transition_function()

        # Check that all states have reward mappings
        assert 0 in reward_fn
        assert 1 in reward_fn

        # Check that reward functions are callable
        for _state, transitions in reward_fn.items():
            for _expr, func in transitions.items():
                assert callable(func)

                # Test that reward functions can be called
                reward = func(None, None, None)
                assert isinstance(reward, (int, float))

    def test_reward_machine_terminal_state_handling(self):
        """Test that reward machine handles terminal states correctly."""
        rm = ConcreteRewardMachine()
        state_fn = rm._get_state_transition_function()

        # Check that terminal state is properly defined
        assert state_fn[1]["EVENT_B"] == -1  # Terminal state

    def test_reward_machine_no_counter_requirement(self):
        """Test that reward machines don't require counter-related methods."""
        rm = ConcreteRewardMachine()

        # Should not have c_0 property (that's for CountingRewardMachine)
        assert not hasattr(rm, "c_0")

        # Should not have counter-related methods
        assert not hasattr(rm, "_get_counter_transition_function")
        assert not hasattr(rm, "sample_counter_configurations")


class TestRewardMachineIntegration:
    """Test RewardMachine integration with other components."""

    def test_reward_machine_with_adapter(self):
        """Test that RewardMachine works correctly with RmToCrmAdapter."""
        rm = ConcreteRewardMachine()
        adapter = RmToCrmAdapter(rm)

        # Adapter should be properly initialized
        assert adapter._rm is rm
        assert adapter.env_prop_enum is rm.env_prop_enum
        assert adapter.u_0 == rm.u_0

    def test_adapter_preserves_rm_behavior(self):
        """Test that the adapter preserves the behavior of the underlying RM."""
        rm = ConcreteRewardMachine()
        adapter = RmToCrmAdapter(rm)

        # Test a transition
        u_next, c_next, reward_fn = adapter.transition(0, (0,), {EnvProps.EVENT_A})

        # Should transition to state 1 (from RM logic)
        assert u_next == 1
        assert c_next == (0,)  # Adapter always uses (0,) counter
        assert callable(reward_fn)

        # Test reward function execution
        reward = reward_fn(None, None, None)
        assert reward == 1  # From RM reward function for EVENT_A

    def test_adapter_handles_terminal_states_from_rm(self):
        """Test that the adapter correctly handles terminal states from the RM."""
        rm = ConcreteRewardMachine()
        adapter = RmToCrmAdapter(rm)

        # Transition to terminal state
        u_next, c_next, reward_fn = adapter.transition(1, (0,), {EnvProps.EVENT_B})

        # Should be terminal state
        assert u_next == adapter.F[0]  # Should be the terminal state
        assert c_next == (0,)
        assert callable(reward_fn)

        # Test reward function
        reward = reward_fn(None, None, None)
        assert reward == 10  # From RM terminal state reward

    def test_adapter_with_different_rm_configurations(self):
        """Test that the adapter works with different RM configurations."""

        # Create RM with different configuration
        class DifferentRewardMachine(RewardMachine):
            @property
            def u_0(self) -> int:
                return 5  # Different initial state

            def _get_state_transition_function(self) -> dict:
                return {
                    5: {"EVENT_A": 10, "not EVENT_A": 5},
                    10: {"EVENT_B": -1, "not EVENT_B": 10},
                }

            def _get_reward_transition_function(self) -> dict:
                return {
                    5: {
                        "EVENT_A": lambda obs, action, next_obs: 5,
                        "not EVENT_A": lambda obs, action, next_obs: 0,
                    },
                    10: {
                        "EVENT_B": lambda obs, action, next_obs: 100,
                        "not EVENT_B": lambda obs, action, next_obs: 0,
                    },
                }

        different_rm = DifferentRewardMachine(env_prop_enum=EnvProps)
        adapter = RmToCrmAdapter(different_rm)

        # Test that adapter preserves the different RM's behavior
        assert adapter.u_0 == 5

        # Test transition
        u_next, c_next, reward_fn = adapter.transition(5, (0,), {EnvProps.EVENT_A})
        assert u_next == 10
        assert c_next == (0,)
        assert reward_fn(None, None, None) == 5


class TestRewardMachineErrorHandling:
    """Test error handling in RewardMachine implementations."""

    def test_reward_machine_missing_u_0_property(self):
        """Test that RewardMachine without u_0 property raises appropriate error."""

        class IncompleteRewardMachine(RewardMachine):
            def _get_state_transition_function(self) -> dict:
                return {0: {"EVENT_A": 1}}

            def _get_reward_transition_function(self) -> dict:
                return {0: {"EVENT_A": lambda obs, action, next_obs: 0}}

            # Missing u_0 property

        with pytest.raises(
            TypeError,
            match=r"Can't instantiate abstract class.*"
            r"(?:with|without an implementation for) abstract method",
        ):
            IncompleteRewardMachine(env_prop_enum=EnvProps)

    def test_reward_machine_missing_state_transition_function(self):
        """Test that RewardMachine without state transition function raises error."""

        class IncompleteRewardMachine(RewardMachine):
            @property
            def u_0(self) -> int:
                return 0

            def _get_reward_transition_function(self) -> dict:
                return {0: {"EVENT_A": lambda obs, action, next_obs: 0}}

            # Missing _get_state_transition_function

        with pytest.raises(
            TypeError,
            match="Can't instantiate abstract class.*_get_state_transition_function",
        ):
            IncompleteRewardMachine(env_prop_enum=EnvProps)

    def test_reward_machine_missing_reward_transition_function(self):
        """Test that RewardMachine without reward transition function raises error."""

        class IncompleteRewardMachine(RewardMachine):
            @property
            def u_0(self) -> int:
                return 0

            def _get_state_transition_function(self) -> dict:
                return {0: {"EVENT_A": 1}}

            # Missing _get_reward_transition_function

        with pytest.raises(
            TypeError,
            match="Can't instantiate abstract class.*_get_reward_transition_function",
        ):
            IncompleteRewardMachine(env_prop_enum=EnvProps)
