import numpy as np
from panda_gym.pybullet import PyBullet

import experiments.warehouse.constants.environment as ce
import experiments.warehouse.constants.simulation as cs
import experiments.warehouse.constants.waypoints as cw
from experiments.warehouse.scenes.basic import BasicSceneConstructor


class FancySceneConstructor(BasicSceneConstructor):
    """Construct the full version of the warehouse scene."""

    def __init__(
        self,
        sim: PyBullet,
        show_ee_identifier: bool = False,
        show_waypoints: bool = False,
        show_regions: bool = False,
    ) -> None:
        """Initialise the scene constructor."""
        self.sim = sim
        self.show_ee_identifier = show_ee_identifier
        self.show_waypoints = show_waypoints
        self.show_regions = show_regions

    def construct(self) -> None:
        """Construct the scene."""
        self._setup_environment()
        self._setup_cubes()
        self._setup_conveyors()
        self._setup_tray()

        if self.show_ee_identifier:
            self._setup_ee_pos_identifier()
        if self.show_waypoints:
            self._setup_waypoints()
        if self.show_regions:
            self._setup_regions()

    def _setup_environment(self) -> None:
        """Setup the environment."""
        self.sim.create_plane(z_offset=-0.5)
        self.sim.create_table(length=0.4, height=0.5, width=0.4, x_offset=-0.15)

    def _setup_conveyors(self) -> None:
        """Setup the conveyors."""
        self._create_conveyor(x=cs.GREEN_CUBE_POSITION[0])
        self._create_conveyor(x=cs.RED_CUBE_POSITION[0])
        self._create_conveyor(x=cs.BLUE_CUBE_POSITION[0])
        self._create_rail(x=cs.RED_CUBE_POSITION[0] + 0.14 / 2)
        self._create_rail(x=cs.GREEN_CUBE_POSITION[0] + 0.14 / 2)
        self._create_rail(x=cs.BLUE_CUBE_POSITION[0] + 0.14 / 2)
        self._create_rail(x=cs.RED_CUBE_POSITION[0] - 0.14 / 2)
        self._create_tray_conveyor()

    def _create_conveyor(self, x) -> None:
        """Create a conveyor belt."""
        for i in range(0, 18):
            cylinder_pos = np.array([x, -0.25, -0.01]) + np.array([0.0, 0.025, 0.0]) * i
            cylinder_orientation = np.array([0, 1, 0, 1])
            self.sim.create_cylinder(
                body_name=f"cylinder_{i}",
                radius=0.01,
                height=0.14,
                mass=0.0,
                position=cylinder_pos,
                rgba_color=ce.CONVEYOR_BELT_COLOR,
            )
            # Rotate 90 degrees around x axis
            self.sim.set_base_pose(
                body=f"cylinder_{i}",
                position=cylinder_pos,
                orientation=cylinder_orientation,
            )

    def _create_rail(self, x) -> None:
        """Create a rail."""
        self.sim.create_box(
            body_name="rail_one",
            half_extents=cs.RAIL_HALF_EXTENTS,
            mass=0.0,
            ghost=True,
            position=np.array([x, -0.035, -0.01]),
            rgba_color=ce.RAIL_COLOR,
        )

    def _create_tray_conveyor(self) -> None:
        """Create the tray conveyor."""
        for i in range(0, 30):
            cylinder_pos = (
                np.array(
                    [
                        cw.RELEASE_GREEN[0] - 0.3,
                        cw.RELEASE_GREEN[1] + 0.05,
                        -0.2,
                    ]
                )
                + np.array([0.025, 0.00, 0.0]) * i
            )
            cylinder_orientation = np.array([1, 0, 0, 1])

            self.sim.create_cylinder(
                body_name=f"cylinder_{i}",
                radius=0.01,
                height=0.3,
                mass=0.0,
                position=cylinder_pos,
                rgba_color=ce.CONVEYOR_BELT_COLOR,
            )
            self.sim.set_base_pose(
                body=f"cylinder_{i}",
                position=cylinder_pos,
                orientation=cylinder_orientation,
            )

        self.sim.create_box(
            body_name="large_rail_one",
            half_extents=cs.LARGE_RAIL_HALF_EXTENTS,
            mass=0.0,
            ghost=True,
            position=cs.LARGE_RAIL_ONE_POSITION,
            rgba_color=ce.RAIL_COLOR,
        )
        self.sim.create_box(
            body_name="large_rail_two",
            half_extents=cs.LARGE_RAIL_HALF_EXTENTS,
            mass=0.0,
            ghost=True,
            position=cs.LARGE_RAIL_TWO_POSITION,
            rgba_color=ce.RAIL_COLOR,
        )

    def _setup_tray(self) -> None:
        """Setup the tray."""
        self.sim.loadURDF(
            body_name="tray",
            **{
                "fileName": "tray/tray.urdf",
                "basePosition": cs.TRAY_POSITION_GREEN,
                "baseOrientation": self.sim.physics_client.getQuaternionFromEuler(
                    [0, 0, -1 * np.pi / 2]
                ),
                "globalScaling": 0.5,
                "useFixedBase": False,
            },
        )
