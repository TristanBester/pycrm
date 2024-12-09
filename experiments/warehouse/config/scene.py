import numpy as np

import experiments.warehouse.constants.regions as cr
import experiments.warehouse.constants.simulation as cs
import experiments.warehouse.constants.waypoints as cw


def get_cube_configs() -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return the configuration for the cubes in the scene."""
    return [
        ("red", cs.RED_CUBE_POSITION, np.array([1, 0, 0, 0.8])),
        ("green", cs.GREEN_CUBE_POSITION, np.array([0, 1, 0, 0.8])),
        ("blue", cs.BLUE_CUBE_POSITION, np.array([0, 1, 1, 0.8])),
    ]


def get_waypoint_configs() -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Return the configuration for the waypoints in the scene."""
    return [
        ("grasp", cw.GRASP_RED, cw.GRASP_GREEN, cw.GRASP_BLUE),
        ("above", cw.ABOVE_RED, cw.ABOVE_GREEN, cw.ABOVE_BLUE),
        ("release", cw.RELEASE_RED, cw.RELEASE_GREEN, cw.RELEASE_BLUE),
    ]


def get_region_configs() -> list[tuple[str, np.ndarray, list[np.ndarray], np.ndarray]]:
    """Return the configuration for the regions in the scene."""
    return [
        (
            "safe_region",
            cr.SAFE_REGION_HALF_EXTENTS,
            [
                cr.SAFE_REGION_RED_COM,
                cr.SAFE_REGION_GREEN_COM,
                cr.SAFE_REGION_BLUE_COM,
            ],
            np.array([1, 0, 1, 0.2]),
        ),
        (
            "tight_region",
            cr.TIGHT_REGION_HALF_EXTENTS,
            [
                cr.TIGHT_REGION_RED_COM,
                cr.TIGHT_REGION_GREEN_COM,
                cr.TIGHT_REGION_BLUE_COM,
            ],
            np.array([1, 1, 0, 0.2]),
        ),
        (
            "release_region",
            cr.RELEASE_REGION_HALF_EXTENTS,
            [
                cr.RELEASE_REGION_RED_COM,
                cr.RELEASE_REGION_GREEN_COM,
                cr.RELEASE_REGION_BLUE_COM,
            ],
            np.array([1, 0, 1, 0.2]),
        ),
    ]
