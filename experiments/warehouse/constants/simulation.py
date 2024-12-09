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
