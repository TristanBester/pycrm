from itertools import product

from examples.crm.tabular.core.label import Symbol
from pycrm.automaton import CountingRewardMachine


class OfficeWorldCountingRewardMachine(CountingRewardMachine):
    """Counting reward machine for the Office World environment."""

    def __init__(self):
        """Initialise the counting reward machine."""
        super().__init__(env_prop_enum=Symbol)

    @property
    def u_0(self) -> int:
        """Return the initial state of the machine."""
        return 0

    @property
    def c_0(self) -> tuple[int, ...]:
        """Return the initial counter configuration of the machine."""
        return (0, 0)

    @property
    def encoded_configuration_size(self) -> int:
        """Return the size of the encoded counter configuration."""
        return 3

    def _get_state_transition_function(self) -> dict:
        """Return the state transition function."""
        return {
            0: {
                "M / (-,-)": 0,
                "E / (-,-)": 1,
                "C / (-,-)": 0,
                "P / (-,-)": 0,
                "D / (-,-)": -1,
                "/ (-,-)": 0,
            },
            1: {
                "M / (-,-)": 1,
                "E / (-,-)": -1,
                "C / (-,-)": 1,
                "P / (-,-)": 1,
                "D / (-,-)": -1,
                "/ (NZ,-)": 1,
                "/ (Z,-)": 2,
            },
            2: {
                "M / (-,-)": 2,
                "E / (-,-)": 2,
                "C / (-,-)": 2,
                "P / (-,-)": 2,
                "D / (-,-)": -1,
                "/ (-,NZ)": 2,
                "/ (-,Z)": -1,
            },
        }

    def _get_counter_transition_function(self) -> dict:
        """Return the counter transition function."""
        return {
            0: {
                "M / (-,-)": (1, 1),
                "E / (-,-)": (0, 0),
                "C / (-,-)": (0, 0),
                "P / (-,-)": (0, 0),
                "D / (-,-)": (0, 0),
                "/ (-,-)": (0, 0),
            },
            1: {
                "M / (-,-)": (0, 0),
                "E / (-,-)": (0, 0),
                "C / (-,-)": (-1, 0),
                "P / (-,-)": (0, 0),
                "D / (-,-)": (0, 0),
                "/ (NZ,-)": (0, 0),
                "/ (Z,-)": (0, 0),
            },
            2: {
                "M / (-,-)": (0, 0),
                "E / (-,-)": (0, 0),
                "C / (-,-)": (0, 0),
                "P / (-,-)": (0, -1),
                "D / (-,-)": (0, -1),
                "/ (-,NZ)": (0, 0),
                "/ (-,Z)": (0, 0),
            },
        }

    def _get_reward_transition_function(self) -> dict:
        """Return the reward transition function."""
        return {
            0: {
                "M / (-,-)": -0.1,
                "E / (-,-)": -0.1,
                "C / (-,-)": -0.1,
                "P / (-,-)": -0.1,
                "D / (-,-)": -100,
                "/ (-,-)": -0.1,
            },
            1: {
                "M / (-,-)": -0.1,
                "E / (-,-)": -100,
                "C / (-,-)": -0.1,
                "P / (-,-)": -0.1,
                "D / (-,-)": -100,
                "/ (NZ,-)": -0.1,
                "/ (Z,-)": -0.1,
            },
            2: {
                "M / (-,-)": -0.1,
                "E / (-,-)": -0.1,
                "C / (-,-)": -0.1,
                "P / (-,-)": -0.1,
                "D / (-,-)": -100,
                "/ (-,NZ)": -0.1,
                "/ (-,Z)": 100.0,
            },
        }

    def _get_possible_counter_configurations(self) -> list[tuple[int, ...]]:
        """Return the possible counter configurations."""
        return list(product(range(3), repeat=2))

    def sample_counter_configurations(self) -> list[tuple[int, ...]]:
        """Return a sample counter configuration."""
        return self._get_possible_counter_configurations()
