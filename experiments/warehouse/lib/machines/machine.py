from collections import defaultdict
from itertools import product

import numpy as np

from crm.automaton import CountingRewardMachine
from experiments.warehouse.lib.label import WarehouseEvent
from experiments.warehouse.lib.machines.reward import recaled_reward_function
from experiments.warehouse.lib.machines.stage import (
    create_above_block_stage,
    create_drop_stage,
    create_grasp_stage,
    create_grip_stage,
    create_release_position_stage,
)


class ContextSensitiveCRM(CountingRewardMachine):
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
        return (3, 3, 3)

    def sample_counter_configurations(self) -> list[tuple[int, ...]]:
        """Return a list of counter configurations."""
        main_configurations = list(product(range(3), repeat=3))

        random_configurations = np.random.randint(low=0, high=100_000, size=(15, 3))
        random_configurations[5:, 0] = 0
        random_configurations[10:, 1] = 0
        random_configurations = [tuple(config) for config in random_configurations]

        configurations = main_configurations + random_configurations
        return configurations

    def _get_state_transition_function(self) -> dict:
        """Return the state transition function."""
        decision_node_state_transitions = {
            0: {
                "/ (Z, Z, Z)": -1,
                "/ (NZ, -, -)": 15,
                "/ (Z, NZ, -)": 10,
                "/ (Z, Z, NZ)": 5,
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
                "/ (Z, Z, Z)": (0, 0, 0),
                "/ (NZ, -, -)": (0, 0, 0),
                "/ (Z, NZ, -)": (0, 0, 0),
                "/ (Z, Z, NZ)": (0, 0, 0),
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
                "/ (Z, Z, Z)": 0.0,
                "/ (NZ, -, -)": 0.0,
                "/ (Z, NZ, -)": 0.0,
                "/ (Z, Z, NZ)": 0.0,
            }
        }

        # We have a set of reward functions each in the range [0, 1]
        # Now need to rescale them to ensure monotonicity
        block_module_reward_transitions = defaultdict(dict)
        for t in self._transitions:
            if t.reward_fn._enable_rescaling:
                t.reward_fn._intercept = -t.current_state
                t.reward_fn._min_reward = -t.current_state - 1
                t.reward_fn._max_reward = -t.current_state
                t.reward_fn = recaled_reward_function(
                    rf=t.reward_fn,
                    r_min=-10.0,
                    r_max=1.0,
                )
                t.reward_fn._intercept = -t.current_state / 10 - 1.0

            block_module_reward_transitions[t.current_state][t.formula] = t.reward_fn

        x = decision_node_reward_transitions | dict(block_module_reward_transitions)
        return x

    def _init_transitions(self) -> None:
        """Add the block module to the state transition function."""
        self._transitions = []

        ######
        # RED
        ######

        # Move above red block
        self._transitions += create_above_block_stage(
            block_colour="RED",
            counter_state="(NZ,-,-)",
            current_state=15,
            success_state=14,
            base_counter_modifier=(0, 0, 0),
            success_counter_modifier=(0, 0, 0),
        )
        # Move to red block grasp position
        self._transitions += create_grasp_stage(
            block_colour="RED",
            counter_state="(NZ,-,-)",
            current_state=14,
            success_state=13,
            base_counter_modifier=(0, 0, 0),
            success_counter_modifier=(0, 0, 0),
        )
        # Grasp red block
        self._transitions += create_grip_stage(
            block_colour="RED",
            counter_state="(NZ,-,-)",
            current_state=13,
            success_state=12,
            base_counter_modifier=(0, 0, 0),
            success_counter_modifier=(0, 0, 0),
        )
        # Move to red block release position
        self._transitions += create_release_position_stage(
            block_colour="RED",
            counter_state="(NZ,-,-)",
            current_state=12,
            success_state=11,
            base_counter_modifier=(0, 0, 0),
            success_counter_modifier=(0, 0, 0),
        )
        # Release red block
        self._transitions += create_drop_stage(
            block_colour="RED",
            counter_state="(NZ,-,-)",
            current_state=11,
            success_state=0,
            base_counter_modifier=(0, 0, 0),
            success_counter_modifier=(-1, 0, 0),
        )

        ######
        # GREEN
        ######

        # Move aboove green block
        self._transitions += create_above_block_stage(
            block_colour="GREEN",
            counter_state="(Z,NZ,-)",
            current_state=10,
            success_state=9,
            base_counter_modifier=(0, 0, 0),
            success_counter_modifier=(0, 0, 0),
        )
        # Move to green block grasp position
        self._transitions += create_grasp_stage(
            block_colour="GREEN",
            counter_state="(Z,NZ,-)",
            current_state=9,
            success_state=8,
            base_counter_modifier=(0, 0, 0),
            success_counter_modifier=(0, 0, 0),
        )
        # Grasp green block
        self._transitions += create_grip_stage(
            block_colour="GREEN",
            counter_state="(Z,NZ,-)",
            current_state=8,
            success_state=7,
            base_counter_modifier=(0, 0, 0),
            success_counter_modifier=(0, 0, 0),
        )
        # Move to green block release position
        self._transitions += create_release_position_stage(
            block_colour="GREEN",
            counter_state="(Z,NZ,-)",
            current_state=7,
            success_state=6,
            base_counter_modifier=(0, 0, 0),
            success_counter_modifier=(0, 0, 0),
        )
        # Release green block
        self._transitions += create_drop_stage(
            block_colour="GREEN",
            counter_state="(Z,NZ,-)",
            current_state=6,
            success_state=0,
            base_counter_modifier=(0, 0, 0),
            success_counter_modifier=(0, -1, 0),
        )

        ######
        # BLUE
        ######

        # Move aboove blue block
        self._transitions += create_above_block_stage(
            block_colour="BLUE",
            counter_state="(Z,Z,NZ)",
            current_state=5,
            success_state=4,
            base_counter_modifier=(0, 0, 0),
            success_counter_modifier=(0, 0, 0),
        )
        # Move to green block grasp position
        self._transitions += create_grasp_stage(
            block_colour="BLUE",
            counter_state="(Z,Z,NZ)",
            current_state=4,
            success_state=3,
            base_counter_modifier=(0, 0, 0),
            success_counter_modifier=(0, 0, 0),
        )
        # Grasp blue block
        self._transitions += create_grip_stage(
            block_colour="BLUE",
            counter_state="(Z,Z,NZ)",
            current_state=3,
            success_state=2,
            base_counter_modifier=(0, 0, 0),
            success_counter_modifier=(0, 0, 0),
        )
        # Move to blue block release position
        self._transitions += create_release_position_stage(
            block_colour="BLUE",
            counter_state="(Z,Z,NZ)",
            current_state=2,
            success_state=1,
            base_counter_modifier=(0, 0, 0),
            success_counter_modifier=(0, 0, 0),
        )
        # Release blue block
        self._transitions += create_drop_stage(
            block_colour="BLUE",
            counter_state="(Z,Z,NZ)",
            current_state=1,
            success_state=0,
            base_counter_modifier=(0, 0, 0),
            success_counter_modifier=(0, 0, -1),
        )
