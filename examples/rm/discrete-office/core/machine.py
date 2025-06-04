from crm.automaton import RewardMachine
from examples.rm.discrete.core.label import Symbol


class OfficeWorldRewardMachine(RewardMachine):
    """Reward machine for the Office World environment."""

    def __init__(self):
        """Initialise the counting reward machine."""
        super().__init__(env_prop_enum=Symbol)

    @property
    def u_0(self) -> int:
        """Return the initial state of the machine."""
        return 0

    @property
    def encoded_configuration_size(self) -> int:
        """Return the size of the encoded counter configuration."""
        return 4

    def _get_state_transition_function(self) -> dict:
        """Return the state transition function."""
        return {
            0: {
                "M": 1,
                "DEFAULT": 0,
            },
            1: {
                "C": 2,
                "DEFAULT": 1,
            },
            2: {
                "P": -1,
                "DEFAULT": 2,
            },
        }

    def _get_reward_transition_function(self) -> dict:
        """Return the reward transition function."""
        return {
            0: {
                "M": 1,
                "DEFAULT": 0,
            },
            1: {
                "C": 1,
                "DEFAULT": 0,
            },
            2: {
                "P": 1,
                "DEFAULT": 0,
            },
        }
