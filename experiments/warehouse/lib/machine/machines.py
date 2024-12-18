from collections import defaultdict
from itertools import product

from crm.automaton import CountingRewardMachine
from experiments.warehouse.lib.label import WarehouseEvent
from experiments.warehouse.lib.machine.stage import (
    create_above_block_stage,
    create_drop_stage,
    create_grasp_stage,
    create_grip_stage,
    create_release_position_stage,
)


class RegularCRM(CountingRewardMachine):
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
        return (3,)

    def sample_counter_configurations(self) -> list[tuple[int]]:
        """Return a list of counter configurations."""
        return [(3,), (2,), (1,)]

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
                "/ (Z)": 10.0,
                "/ (NZ)": 0.0,
            }
        }

        block_module_reward_transitions = defaultdict(dict)
        for t in self._transitions:
            t.reward_fn._intercept = -t.current_state
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
            remain_counter_modifier=(0,),
            progress_counter_modifier=(0,),
        )
        # Move to grasp position
        self._transitions += create_grasp_stage(
            block_colour=block_colour,
            counter_state=counter_state,
            current_state=4,
            success_state=3,
            remain_counter_modifier=(0,),
            progress_counter_modifier=(0,),
        )
        # Grasp block
        self._transitions += create_grip_stage(
            block_colour=block_colour,
            counter_state=counter_state,
            current_state=3,
            success_state=2,
            remain_counter_modifier=(0,),
            progress_counter_modifier=(0,),
        )
        # Move to release position
        self._transitions += create_release_position_stage(
            block_colour=block_colour,
            counter_state=counter_state,
            current_state=2,
            success_state=1,
            remain_counter_modifier=(0,),
            progress_counter_modifier=(0,),
        )
        # Release block
        self._transitions += create_drop_stage(
            block_colour=block_colour,
            counter_state=counter_state,
            current_state=1,
            success_state=0,
            remain_counter_modifier=(0,),
            progress_counter_modifier=(-1,),
        )


class ContextFreeCRM(CountingRewardMachine):
    """Context-free counting reward machine for the warehouse environment."""

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
        return (3, 3)

    def sample_counter_configurations(self) -> list[tuple[int, ...]]:
        """Return a list of counter configurations."""
        return list(product(range(4), repeat=2))

    def _get_state_transition_function(self) -> dict:
        """Return the state transition function."""
        decision_node_state_transitions = {
            0: {
                "/ (Z, Z)": -1,
                "/ (NZ, -)": 10,
                "/ (Z, NZ)": 5,
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
                "/ (Z, Z)": (0, 0),
                "/ (NZ, -)": (0, 0),
                "/ (Z, NZ)": (0, 0),
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
                "/ (Z, Z)": 0.0,
                "/ (NZ, -)": 0.0,
                "/ (Z, NZ)": 0.0,
            }
        }

        block_module_reward_transitions = defaultdict(dict)
        for t in self._transitions:
            t.reward_fn._intercept = -t.current_state
            block_module_reward_transitions[t.current_state][t.formula] = t.reward_fn

        x = decision_node_reward_transitions | dict(block_module_reward_transitions)
        return x

    def _init_transitions(self) -> None:
        """Add the block module to the state transition function."""
        self._transitions = []

        # Move above red block
        self._transitions += create_above_block_stage(
            block_colour="RED",
            counter_state="(NZ,-)",
            current_state=10,
            success_state=9,
            remain_counter_modifier=(0, 0),
            progress_counter_modifier=(0, 0),
        )
        # Move to red blockgrasp position
        self._transitions += create_grasp_stage(
            block_colour="RED",
            counter_state="(NZ,-)",
            current_state=9,
            success_state=8,
            remain_counter_modifier=(0, 0),
            progress_counter_modifier=(0, 0),
        )
        # Grasp red block
        self._transitions += create_grip_stage(
            block_colour="RED",
            counter_state="(NZ,-)",
            current_state=8,
            success_state=7,
            remain_counter_modifier=(0, 0),
            progress_counter_modifier=(0, 0),
        )
        # Move to red block release position
        self._transitions += create_release_position_stage(
            block_colour="RED",
            counter_state="(NZ,-)",
            current_state=7,
            success_state=6,
            remain_counter_modifier=(0, 0),
            progress_counter_modifier=(0, 0),
        )
        # Release red block
        self._transitions += create_drop_stage(
            block_colour="RED",
            counter_state="(NZ,-)",
            current_state=6,
            success_state=0,
            remain_counter_modifier=(0, 0),
            progress_counter_modifier=(-1, 0),
        )

        # Move above green block
        self._transitions += create_above_block_stage(
            block_colour="GREEN",
            counter_state="(Z,NZ)",
            current_state=5,
            success_state=4,
            remain_counter_modifier=(0, 0),
            progress_counter_modifier=(0, 0),
        )
        # Move to green block grasp position
        self._transitions += create_grasp_stage(
            block_colour="GREEN",
            counter_state="(Z,NZ)",
            current_state=4,
            success_state=3,
            remain_counter_modifier=(0, 0),
            progress_counter_modifier=(0, 0),
        )
        # Grasp green block
        self._transitions += create_grip_stage(
            block_colour="GREEN",
            counter_state="(Z,NZ)",
            current_state=3,
            success_state=2,
            remain_counter_modifier=(0, 0),
            progress_counter_modifier=(0, 0),
        )
        # Move to green block release position
        self._transitions += create_release_position_stage(
            block_colour="GREEN",
            counter_state="(Z,NZ)",
            current_state=2,
            success_state=1,
            remain_counter_modifier=(0, 0),
            progress_counter_modifier=(0, 0),
        )
        # Release green block
        self._transitions += create_drop_stage(
            block_colour="GREEN",
            counter_state="(Z,NZ)",
            current_state=1,
            success_state=0,
            remain_counter_modifier=(0, 0),
            progress_counter_modifier=(0, -1),
        )


class ContextSensitiveCRM(CountingRewardMachine):
    """Context-sensitive counting reward machine for the warehouse environment."""

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
        return (3, 3, 3)

    def sample_counter_configurations(self) -> list[tuple[int, ...]]:
        """Return a list of counter configurations."""
        return list(product(range(4), repeat=3))

    def _get_state_transition_function(self) -> dict:
        """Return the state transition function."""
        decision_node_state_transitions = {
            0: {
                # All blocks packed
                "/ (Z, Z, Z)": -1,
                # Pack green blocks
                "/ (NZ, -, -)": 15,
                # Pack red blocks
                "/ (Z, NZ, -)": 10,
                # Pack blue blocks
                "/ (Z, Z, NZ)": 5,
            },
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
                "/ (Z, Z, Z)": (0, 0, 0),
                "/ (NZ, -, -)": (0, 0, 0),
                "/ (Z, NZ, -)": (0, 0, 0),
                "/ (Z, Z, NZ)": (0, 0, 0),
            },
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
                "/ (Z, Z, Z)": 0.0,
                "/ (NZ, -, -)": 0.0,
                "/ (Z, NZ, -)": 0.0,
                "/ (Z, Z, NZ)": 0.0,
            }
        }

        block_module_reward_transitions = defaultdict(dict)
        for t in self._transitions:
            t.reward_fn._intercept = -t.current_state
            block_module_reward_transitions[t.current_state][t.formula] = t.reward_fn

        x = decision_node_reward_transitions | dict(block_module_reward_transitions)
        return x

    def _init_transitions(self) -> None:
        """Add the block module to the state transition function."""
        self._transitions = []

        # Move above red block
        self._transitions += create_above_block_stage(
            block_colour="RED",
            counter_state="(NZ,-, -)",
            current_state=15,
            success_state=14,
            remain_counter_modifier=(0, 0, 0),
            progress_counter_modifier=(0, 0, 0),
        )
        # Move to red blockgrasp position
        self._transitions += create_grasp_stage(
            block_colour="RED",
            counter_state="(NZ,-, -)",
            current_state=14,
            success_state=13,
            remain_counter_modifier=(0, 0, 0),
            progress_counter_modifier=(0, 0, 0),
        )
        # Grasp red block
        self._transitions += create_grip_stage(
            block_colour="RED",
            counter_state="(NZ,-, -)",
            current_state=13,
            success_state=12,
            remain_counter_modifier=(0, 0, 0),
            progress_counter_modifier=(0, 0, 0),
        )
        # Move to red block release position
        self._transitions += create_release_position_stage(
            block_colour="RED",
            counter_state="(NZ,-, -)",
            current_state=12,
            success_state=11,
            remain_counter_modifier=(0, 0, 0),
            progress_counter_modifier=(0, 0, 0),
        )
        # Release red block
        self._transitions += create_drop_stage(
            block_colour="RED",
            counter_state="(NZ,-,-)",
            current_state=11,
            success_state=0,
            remain_counter_modifier=(0, 0, 0),
            progress_counter_modifier=(-1, 0, 0),
        )

        # Move above green block
        self._transitions += create_above_block_stage(
            block_colour="GREEN",
            counter_state="(Z,NZ,-)",
            current_state=10,
            success_state=9,
            remain_counter_modifier=(0, 0, 0),
            progress_counter_modifier=(0, 0, 0),
        )
        # Move to green block grasp position
        self._transitions += create_grasp_stage(
            block_colour="GREEN",
            counter_state="(Z,NZ,-)",
            current_state=9,
            success_state=8,
            remain_counter_modifier=(0, 0, 0),
            progress_counter_modifier=(0, 0, 0),
        )
        # Grasp green block
        self._transitions += create_grip_stage(
            block_colour="GREEN",
            counter_state="(Z,NZ,-)",
            current_state=8,
            success_state=7,
            remain_counter_modifier=(0, 0, 0),
            progress_counter_modifier=(0, 0, 0),
        )
        # Move to green block release position
        self._transitions += create_release_position_stage(
            block_colour="GREEN",
            counter_state="(Z,NZ)",
            current_state=7,
            success_state=6,
            remain_counter_modifier=(0, 0, 0),
            progress_counter_modifier=(0, 0, 0),
        )
        # Release green block
        self._transitions += create_drop_stage(
            block_colour="GREEN",
            counter_state="(Z,NZ,-)",
            current_state=6,
            success_state=0,
            remain_counter_modifier=(0, 0, 0),
            progress_counter_modifier=(0, -1, 0),
        )

        # Move above blue block
        self._transitions += create_above_block_stage(
            block_colour="BLUE",
            counter_state="(Z,Z,NZ)",
            current_state=5,
            success_state=4,
            remain_counter_modifier=(0, 0, 0),
            progress_counter_modifier=(0, 0, 0),
        )
        # Move to blue block grasp position
        self._transitions += create_grasp_stage(
            block_colour="BLUE",
            counter_state="(Z,Z,NZ)",
            current_state=4,
            success_state=3,
            remain_counter_modifier=(0, 0, 0),
            progress_counter_modifier=(0, 0, 0),
        )
        # Grasp blue block
        self._transitions += create_grip_stage(
            block_colour="BLUE",
            counter_state="(Z,Z,NZ)",
            current_state=3,
            success_state=2,
            remain_counter_modifier=(0, 0, 0),
            progress_counter_modifier=(0, 0, 0),
        )
        # Move to blue block release position
        self._transitions += create_release_position_stage(
            block_colour="BLUE",
            counter_state="(Z,Z,NZ)",
            current_state=2,
            success_state=1,
            remain_counter_modifier=(0, 0, 0),
            progress_counter_modifier=(0, 0, 0),
        )
        # Release blue block
        self._transitions += create_drop_stage(
            block_colour="BLUE",
            counter_state="(Z,Z,NZ)",
            current_state=1,
            success_state=0,
            remain_counter_modifier=(0, 0, 0),
            progress_counter_modifier=(0, 0, -1),
        )
