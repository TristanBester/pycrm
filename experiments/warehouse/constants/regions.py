import numpy as np

import experiments.warehouse.constants.simulation as cs
import experiments.warehouse.constants.waypoints as cw

# Region sizes
SAFE_REGION_HALF_EXTENTS = np.array([0.05, 0.05, 0.2]) / 2
TIGHT_REGION_HALF_EXTENTS = np.array([0.05, 0.05, 0.05]) / 1.8
RELEASE_REGION_HALF_EXTENTS = np.array([0.1, 0.1, 0.1]) / 2

# Safe region centres of mass
SAFE_REGION_GREEN_COM = cs.GREEN_CUBE_POSITION + np.array([0, 0, 0.075])
SAFE_REGION_RED_COM = cs.RED_CUBE_POSITION + np.array([0, 0, 0.075])
SAFE_REGION_BLUE_COM = cs.BLUE_CUBE_POSITION + np.array([0, 0, 0.075])

# Tight region centres of mass
TIGHT_REGION_GREEN_COM = cs.GREEN_CUBE_POSITION
TIGHT_REGION_RED_COM = cs.RED_CUBE_POSITION
TIGHT_REGION_BLUE_COM = cs.BLUE_CUBE_POSITION

# Release region centres of mass
RELEASE_REGION_GREEN_COM = cw.RELEASE_GREEN
RELEASE_REGION_RED_COM = cw.RELEASE_RED
RELEASE_REGION_BLUE_COM = cw.RELEASE_BLUE
