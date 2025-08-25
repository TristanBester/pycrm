import inspect

from pycrm.automaton import CountingRewardMachine, RmToCrmAdapter
from tests.conftest import EnvProps


class TestCCRMRewardFunctionReplacement:
    """Test the replacement of the reward functions in the CountingRewardMachine."""

    def test_ccrm_rewards_replaced(self, ccrm: CountingRewardMachine) -> None:
        """Test that the reward functions are replaced with callables."""
        for _, reward_fns in ccrm._delta_r.items():
            for _, reward_fn in reward_fns.items():
                # Test a callable has been created
                assert callable(reward_fn)

                # Test the signature of the callable matches that of a reward function
                signature = inspect.signature(reward_fn)
                assert len(signature.parameters) == 3
                assert list(signature.parameters.keys()) == [
                    "obs",
                    "action",
                    "next_obs",
                ]


class TestAdapterRewardFunctionHandling:
    """Test reward function handling in the RmToCrmAdapter."""

    def test_adapter_preserves_reward_functions(
        self, rm_to_crm_adapter: RmToCrmAdapter
    ):
        """Test that the adapter preserves reward functions from the underlying RM."""
        adapter = rm_to_crm_adapter

        # Get the reward transition function from the adapter
        delta_r = adapter.delta_r

        # Check that all reward functions are callable
        for _state, reward_fns in delta_r.items():
            for _expr, reward_fn in reward_fns.items():
                assert callable(reward_fn)

                # Test the signature
                signature = inspect.signature(reward_fn)
                assert len(signature.parameters) == 3
                assert list(signature.parameters.keys()) == [
                    "obs",
                    "action",
                    "next_obs",
                ]

    def test_adapter_reward_function_execution(self, rm_to_crm_adapter: RmToCrmAdapter):
        """Test that reward functions from the adapter can be executed."""
        adapter = rm_to_crm_adapter

        # Test a transition to get a reward function
        _, _, reward_fn = adapter.transition(0, (0,), set())

        # Execute the reward function
        reward = reward_fn(None, None, None)
        assert isinstance(reward, (int, float))

    def test_adapter_terminal_state_reward(self, rm_to_crm_adapter: RmToCrmAdapter):
        """Test that the adapter handles terminal state rewards correctly."""
        adapter = rm_to_crm_adapter

        # Transition to terminal state
        u_next, _, reward_fn = adapter.transition(1, (0,), {EnvProps.EVENT_B})

        # Should be terminal state
        assert u_next == adapter.F[0]

        # Execute the reward function
        reward = reward_fn(None, None, None)
        assert isinstance(reward, (int, float))
