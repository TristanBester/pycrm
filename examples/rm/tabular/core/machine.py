from examples.rm.tabular.core.label import Symbol
from pycrm.automaton import RewardMachine


class OfficeWorldRewardMachine(RewardMachine):
    """Reward machine for the Office World environment."""

    def __init__(self):
        """Initialise the reward machine."""
        super().__init__(env_prop_enum=Symbol)

    @property
    def u_0(self) -> int:
        """Return the initial state of the machine."""
        return 0

    def _get_state_transition_function(self) -> dict:
        """Return the state transition function."""
        return {
            0: {
                "M": 1,
                "NOT M": 0,
            },
            1: {
                "C": 2,
                "NOT C": 1,
            },
            2: {
                "P": 3,
                "NOT P": 2,
            },
        }

    def _get_reward_transition_function(self) -> dict:
        """Return the reward transition function."""
        return {
            0: {
                "M": 10,
                "NOT M": -0.1,
            },
            1: {
                "C": 10,
                "NOT C": -0.1,
            },
            2: {
                "P": 10,
                "NOT P": -0.1,
            },
        }
