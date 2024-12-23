import numpy as np

# Cube dimensions
CUBE_X_DIM = 0.05
CUBE_Y_DIM = 0.05
CUBE_Z_DIM = 0.05
CUBE_HALF_EXTENTS = np.array([CUBE_X_DIM / 2, CUBE_Y_DIM / 2, CUBE_Z_DIM / 2])

# Cube grid base positions
GRID_BASE_X = 0.2
GRID_BASE_Y = -0.4

# Cube grid deltas
GRID_DELTA_X = 0.15
GRID_DELTA_Y = 0.2

# Cube position
RED_CUBE_POSITION = np.array(
    [
        0.25,
        0.1,
        0.025,
    ]
)
GREEN_CUBE_POSITION = np.array(
    [
        0.4,
        0.1,
        0.025,
    ]
)
BLUE_CUBE_POSITION = np.array(
    [
        0.55,
        0.1,
        0.025,
    ]
)

# Tray position
TRAY_POSITION_RED = np.array([0.25, 0.4, -0.19])
TRAY_POSITION_GREEN = np.array([0.25 + 0.125, 0.4, -0.19])
TRAY_POSITION_BLUE = np.array([0.25 + 0.25, 0.4, -0.19])
TRAY_POSITION_END = np.array([0.25 + 0.4, 0.4, -0.19])

# Rail half extents
RAIL_HALF_EXTENTS = np.array([0.0025, 0.23, 0.02])

# Large rail half extents
LARGE_RAIL_HALF_EXTENTS = np.array([0.38, 0.0025, 0.02])

# Large rail position
LARGE_RAIL_ONE_POSITION = np.array([0.46, 0.25, -0.2])
LARGE_RAIL_TWO_POSITION = np.array([0.46, 0.25 + 0.3, -0.2])
