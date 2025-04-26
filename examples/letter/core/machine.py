from crm.automaton import CountingRewardMachine
from examples.letter.core.label import Symbol


class LetterWorldCountingRewardMachine(CountingRewardMachine):
    """Counting reward machine for the Letter World environment."""

    def __init__(self) -> None:
        """Initialise the counting reward machine."""
        super().__init__(env_prop_enum=Symbol)

    @property
    def u_0(self) -> int:
        """Return the initial state of the machine."""
        return 0

    @property
    def c_0(self) -> tuple[int, ...]:
        """Return the initial counter configuration of the machine."""
        return (0,)

    @property
    def encoded_configuration_size(self) -> int:
        """Return the size of the encoded counter configuration."""
        return 2

    def _get_state_transition_function(self) -> dict:
        """Return the state transition function."""
        return {
            0: {
                "A / (-)": 0,
                "B / (-)": 1,
                "C / (-)": 0,
                "/ (-)": 0,
            },
            1: {
                "A / (-)": 1,
                "B / (-)": 1,
                "C / (NZ)": 1,
                "C / (Z)": -1,
                "/ (-)": 1,
            },
        }

    def _get_counter_transition_function(self) -> dict:
        """Return the counter transition function."""
        return {
            0: {
                "A / (-)": (1,),
                "B / (-)": (0,),
                "C / (-)": (0,),
                "/ (-)": (0,),
            },
            1: {
                "A / (-)": (0,),
                "B / (-)": (0,),
                "C / (NZ)": (-1,),
                "C / (Z)": (0,),
                "/ (-)": (0,),
            },
        }

    def _get_reward_transition_function(self) -> dict:
        """Return the reward transition function."""
        return {
            0: {
                "A / (-)": -0.1,
                "B / (-)": -0.1,
                "C / (-)": -0.1,
                "/ (-)": -0.1,
            },
            1: {
                "A / (-)": -0.1,
                "B / (-)": -0.1,
                "C / (NZ)": -0.1,
                "C / (Z)": 1,
                "/ (-)": -0.1,
            },
        }

    def _get_possible_counter_configurations(self) -> list[tuple[int]]:
        """Return the possible counter configurations."""
        return [(i,) for i in range(6)]

    def sample_counter_configurations(self) -> list[tuple[int]]:
        """Return a sample counter configuration."""
        return self._get_possible_counter_configurations()
