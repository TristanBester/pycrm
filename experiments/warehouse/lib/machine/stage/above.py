import experiments.warehouse.lib.groundenv.constants.waypoints as cw
from experiments.warehouse.lib.machine.reward import (
    create_constant_reward,
    create_waypoint_reward,
)
from experiments.warehouse.lib.machine.stage.transition import Transition


def create_above_block_stage(
    block_colour: str,
    counter_state: str,
    current_state: int,
    success_state: int,
    remain_counter_modifier: tuple[int, ...],
    progress_counter_modifier: tuple[int, ...],
) -> list[Transition]:
    """Create a stage for moving above a block."""
    if block_colour == "RED":
        waypoint = cw.ABOVE_RED
    elif block_colour == "GREEN":
        waypoint = cw.ABOVE_GREEN
    elif block_colour == "BLUE":
        waypoint = cw.ABOVE_BLUE
    else:
        raise ValueError(f"Invalid block colour: {block_colour}")

    transitions = []
    # Gripper closed action executed
    transitions.append(
        Transition(
            formula=f"not GRIPPER_OPEN_ACTION_EXECUTED / {counter_state}",
            current_state=current_state,
            next_state=current_state,
            counter_modifier=remain_counter_modifier,
            reward_fn=create_constant_reward(-1.0),
        )
    )
    # Above block with low velocity
    transitions.append(
        Transition(
            formula=f"ABOVE_{block_colour} and VELOCITY_LOW / {counter_state}",
            current_state=current_state,
            next_state=success_state,
            counter_modifier=progress_counter_modifier,
            reward_fn=create_constant_reward(100.0),
        )
    )
    # Other
    transitions.append(
        Transition(
            formula=f"not (ABOVE_{block_colour} and VELOCITY_LOW) / {counter_state}",
            current_state=current_state,
            next_state=current_state,
            counter_modifier=remain_counter_modifier,
            reward_fn=create_waypoint_reward(
                waypoint=waypoint,
                max_distance=1.0,
            ),
        )
    )
    return transitions
