import inspect

from crm.automaton import CountingRewardMachine


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
