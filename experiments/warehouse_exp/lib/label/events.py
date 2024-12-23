from enum import Enum, auto


class PickPlaceEvent(Enum):
    """Propositions used to capture high-level events in the environment."""

    """
    Waypoints.
    """
    # Waypoints for positions above cubes
    ABOVE_GREEN = auto()
    ABOVE_RED = auto()
    ABOVE_BLUE = auto()

    # Waypoints for grasping positions
    GRASP_GREEN = auto()
    GRASP_RED = auto()
    GRASP_BLUE = auto()

    # Waypoints for release positions
    RELEASE_GREEN = auto()
    RELEASE_RED = auto()
    RELEASE_BLUE = auto()

    # Debug waypoints
    DEBUG_WAYPOINT_ONE = auto()
    DEBUG_WAYPOINT_TWO = auto()

    """
    Regions.
    """
    # Safe regions for the cubes
    SAFE_REGION_GREEN = auto()
    SAFE_REGION_RED = auto()
    SAFE_REGION_BLUE = auto()

    # Tight regions for the cubes
    TIGHT_REGION_GREEN = auto()
    TIGHT_REGION_RED = auto()
    TIGHT_REGION_BLUE = auto()

    # Release regions
    RELEASE_REGION_GREEN = auto()
    RELEASE_REGION_RED = auto()
    RELEASE_REGION_BLUE = auto()

    """
    Gripper & Velocity.
    """
    GRIPPER_OPEN_ACTION_EXECUTED = auto()
    GRIPPER_CLOSED = auto()
    GRIPPER_OPEN = auto()
    VELOCITY_LOW = auto()
