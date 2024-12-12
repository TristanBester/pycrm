from collections import defaultdict

from crm.automaton import CountingRewardMachine
from experiments.warehouse.lib.label import WarehouseEvent
from experiments.warehouse.lib.machine.stage import (
    create_above_block_stage,
    create_drop_stage,
    create_grasp_stage,
    create_grip_stage,
    create_release_position_stage,
)


class WarehouseCountingRewardMachine(CountingRewardMachine):
    """Counting reward machine for the warehouse environment."""

    def __init__(self) -> None:
        """Initialise the counting reward machine."""
        self._init_transitions()
        super().__init__(env_prop_enum=WarehouseEvent)

    @property
    def u_0(self) -> int:
        """Return the initial state of the machine."""
        return 0

    @property
    def c_0(self) -> tuple[int, ...]:
        """Return the initial counter configuration of the machine."""
        return (1,)

    def sample_counter_configurations(self) -> list[tuple[int]]:
        """Return a list of counter configurations."""
        return [(1,)]

    def _get_state_transition_function(self) -> dict:
        """Return the state transition function."""
        decision_node_state_transitions = {
            0: {
                "/ (Z)": -1,
                "/ (NZ)": 5,
            }
        }

        block_module_transitions = defaultdict(dict)
        for t in self._transitions:
            block_module_transitions[t.current_state][t.formula] = t.next_state

        x = decision_node_state_transitions | dict(block_module_transitions)
        return x

    def _get_counter_transition_function(self) -> dict:
        """Return the counter transition function."""
        decision_node_counter_transitions = {
            0: {
                "/ (Z)": (0,),
                "/ (NZ)": (0,),
            }
        }

        block_module_counter_transitions = defaultdict(dict)
        for t in self._transitions:
            block_module_counter_transitions[t.current_state][t.formula] = (
                t.counter_modifier
            )

        x = decision_node_counter_transitions | dict(block_module_counter_transitions)
        return x

    def _get_reward_transition_function(self) -> dict:
        """Return the reward transition function."""
        decision_node_reward_transitions = {
            0: {
                "/ (Z)": 500.0,
                "/ (NZ)": 0.0,
            }
        }

        block_module_reward_transitions = defaultdict(dict)
        for t in self._transitions:
            rf = t.reward_fn
            rf._intercept = -t.current_state
            block_module_reward_transitions[t.current_state][t.formula] = t.reward_fn

        x = decision_node_reward_transitions | dict(block_module_reward_transitions)
        return x

    def _init_transitions(self) -> None:
        """Add the block module to the state transition function."""
        block_colour = "RED"
        counter_state = "(NZ)"

        self._transitions = []
        # Move above block
        self._transitions += create_above_block_stage(
            block_colour=block_colour,
            counter_state=counter_state,
            current_state=5,
            success_state=4,
        )
        # Move to grasp position
        self._transitions += create_grasp_stage(
            block_colour=block_colour,
            counter_state=counter_state,
            current_state=4,
            success_state=3,
        )
        # Grasp block
        self._transitions += create_grip_stage(
            block_colour=block_colour,
            counter_state=counter_state,
            current_state=3,
            success_state=2,
        )
        # Move to release position
        self._transitions += create_release_position_stage(
            block_colour=block_colour,
            counter_state=counter_state,
            current_state=2,
            success_state=1,
        )
        # Release block
        self._transitions += create_drop_stage(
            block_colour=block_colour,
            counter_state=counter_state,
            current_state=1,
            success_state=0,
        )
