import time

import matplotlib.pyplot as plt
import numpy as np
from panda_gym.pybullet import PyBullet

import experiments.warehouse.lib.groundenv.constants.environment as ce
import experiments.warehouse.lib.groundenv.constants.simulation as cs
import experiments.warehouse.lib.groundenv.constants.waypoints as cw
from experiments.warehouse.lib.groundenv.scenes.basic import BasicSceneManager


class FancySceneManager(BasicSceneManager):
    """Construct the full version of the warehouse scene."""

    def __init__(
        self,
        sim: PyBullet,
        show_ee_identifier: bool = False,
        show_waypoints: bool = False,
        show_regions: bool = False,
        frame_delay: float = 0.1,
    ) -> None:
        """Initialise the scene constructor."""
        self.sim = sim
        self.show_ee_identifier = show_ee_identifier
        self.show_waypoints = show_waypoints
        self.show_regions = show_regions
        self._frame_delay = frame_delay

        self._red_success_count = 0
        self._green_success_count = 0
        self._blue_success_count = 0

    @property
    def frame_delay(self) -> float:
        """Get the frame delay for animations in seconds."""
        return self._frame_delay

    def construct(self) -> None:
        """Construct the scene."""
        self._setup_environment()
        self._setup_cubes()
        self._setup_conveyors()
        self._setup_tray()

        self._setup_trajectory()

        if self.show_ee_identifier:
            self._setup_ee_pos_identifier()
        if self.show_waypoints:
            self._setup_waypoints()
        if self.show_regions:
            self._setup_regions()

    def _setup_trajectory(self) -> None:
        """Setup the trajectory."""
        self.trajectory = np.load(
            "/Users/tristan/Projects/counting-reward-machines/experiments/warehouse/exp/cs/utils/trajectory.npy"
        )
        # Take every 10th frame
        # self.trajectory = self.trajectory[::]

        # Create normalized indices from 0 to 1 based on position in array
        normalized_indices = np.linspace(0, 1, len(self.trajectory))

        # Apply colormap to normalized indices to show progression
        colors = plt.get_cmap("jet")(normalized_indices)
        colors = colors[:, :3]  # Keep only RGB values

        # Add 0.5 alpha channel to colors
        colors = np.concatenate([colors, np.full((colors.shape[0], 1), 0.3)], axis=1)

        # 400
        # 900
        # 1350
        # end
        for i in range(len(self.trajectory))[:]:
            self.sim.create_sphere(
                body_name=f"trajectory_{i}",
                radius=0.01,
                mass=0.0,
                ghost=True,
                position=self.trajectory[i],
                rgba_color=colors[i],
            )

    def animate_red_block(self) -> None:
        """Animate the red block."""
        self._red_success_count += 1
        self._translate_red_blocks()

        if self._red_success_count == 3:
            raise ValueError("Fancy scene only supports up to three blocks.")

    def animate_green_block(self) -> None:
        """Animate the green block."""
        self._green_success_count += 1
        self._translate_green_blocks()

        if self._green_success_count == 3:
            raise ValueError("Fancy scene only supports up to three blocks.")

    def animate_blue_block(self) -> None:
        """Animate the blue block."""
        self._blue_success_count += 1
        self._translate_blue_blocks()

        if self._blue_success_count == 3:
            raise ValueError("Fancy scene only supports up to three blocks.")

    def translate_tray(self, destination: str) -> None:
        """Translate the tray to the destination."""
        if destination == "green":
            curr_pos = cs.TRAY_POSITION_RED
            desired_pos = cs.TRAY_POSITION_GREEN
        elif destination == "blue":
            curr_pos = cs.TRAY_POSITION_GREEN
            desired_pos = cs.TRAY_POSITION_BLUE
        elif destination == "end":
            curr_pos = cs.TRAY_POSITION_BLUE
            desired_pos = cs.TRAY_POSITION_END
        else:
            raise ValueError(f"Invalid destination: {destination}")

        for i in range(100):
            self.sim.set_base_pose(
                "tray",
                curr_pos + (desired_pos - curr_pos) * i / 100,
                self.sim.physics_client.getQuaternionFromEuler([0, 0, -1 * np.pi / 2]),
            )
            self.sim.step()
            time.sleep(self.frame_delay)

    def _translate_red_blocks(self) -> None:
        """Translate the red blocks."""
        delta = np.array([0.0, 0.1, 0.0])

        if self._red_success_count == 1:
            red_two_pos = cs.RED_CUBE_POSITION - delta
            red_three_pos = cs.RED_CUBE_POSITION - 2 * delta
        elif self._red_success_count == 2:
            red_two_pos = cs.RED_CUBE_POSITION
            red_three_pos = cs.RED_CUBE_POSITION - delta
        else:
            raise ValueError("Invalid red success count")

        for i in range(20):
            if self._red_success_count == 1:
                self.sim.set_base_pose(
                    "red_two_cube",
                    red_two_pos + delta * i / 20,
                    np.array([0, 0, 0, 1]),
                )
                self.sim.set_base_pose(
                    "red_three_cube",
                    red_three_pos + delta * i / 20,
                    np.array([0, 0, 0, 1]),
                )
            elif self._red_success_count == 2:
                self.sim.set_base_pose(
                    "red_three_cube",
                    red_three_pos + delta * i / 20,
                    np.array([0, 0, 0, 1]),
                )
            self.sim.step()
            time.sleep(self.frame_delay)

    def _translate_green_blocks(self) -> None:
        """Translate the green blocks."""
        delta = np.array([0.0, 0.1, 0.0])

        if self._green_success_count == 1:
            green_two_pos = cs.GREEN_CUBE_POSITION - delta
            green_three_pos = cs.GREEN_CUBE_POSITION - 2 * delta
        elif self._green_success_count == 2:
            green_two_pos = cs.GREEN_CUBE_POSITION
            green_three_pos = cs.GREEN_CUBE_POSITION - delta
        else:
            raise ValueError("Invalid green success count")

        for i in range(20):
            if self._green_success_count == 1:
                self.sim.set_base_pose(
                    "green_two_cube",
                    green_two_pos + delta * i / 20,
                    np.array([0, 0, 0, 1]),
                )
                self.sim.set_base_pose(
                    "green_three_cube",
                    green_three_pos + delta * i / 20,
                    np.array([0, 0, 0, 1]),
                )
            elif self._green_success_count == 2:
                self.sim.set_base_pose(
                    "green_three_cube",
                    green_three_pos + delta * i / 20,
                    np.array([0, 0, 0, 1]),
                )
            self.sim.step()
            time.sleep(self.frame_delay)

    def _translate_blue_blocks(self) -> None:
        """Translate the blue blocks."""
        delta = np.array([0.0, 0.1, 0.0])

        if self._blue_success_count == 1:
            blue_two_pos = cs.BLUE_CUBE_POSITION - delta
            blue_three_pos = cs.BLUE_CUBE_POSITION - 2 * delta
        elif self._blue_success_count == 2:
            blue_two_pos = cs.BLUE_CUBE_POSITION
            blue_three_pos = cs.BLUE_CUBE_POSITION - delta
        else:
            raise ValueError("Invalid blue success count")

        for i in range(20):
            if self._blue_success_count == 1:
                self.sim.set_base_pose(
                    "blue_two_cube",
                    blue_two_pos + delta * i / 20,
                    np.array([0, 0, 0, 1]),
                )
                self.sim.set_base_pose(
                    "blue_three_cube",
                    blue_three_pos + delta * i / 20,
                    np.array([0, 0, 0, 1]),
                )
            elif self._blue_success_count == 2:
                self.sim.set_base_pose(
                    "blue_three_cube",
                    blue_three_pos + delta * i / 20,
                    np.array([0, 0, 0, 1]),
                )
            self.sim.step()
            time.sleep(self.frame_delay)

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
                "basePosition": cs.TRAY_POSITION_RED,
                "baseOrientation": self.sim.physics_client.getQuaternionFromEuler(
                    [0, 0, -1 * np.pi / 2]
                ),
                "globalScaling": 0.5,
                "useFixedBase": False,
            },
        )

    def _get_cube_configs(self) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """Return the configuration for the cubes in the scene."""
        return [
            ("red_one", cs.RED_CUBE_POSITION, np.array([1, 0, 0, 0.8])),
            (
                "red_two",
                cs.RED_CUBE_POSITION + np.array([0.0, -0.1, 0.0]),
                np.array([1, 0, 0, 0.8]),
            ),
            (
                "red_three",
                cs.RED_CUBE_POSITION + np.array([0.0, -0.2, 0.0]),
                np.array([1, 0, 0, 0.8]),
            ),
            ("green_one", cs.GREEN_CUBE_POSITION, np.array([0, 1, 0, 0.8])),
            (
                "green_two",
                cs.GREEN_CUBE_POSITION + np.array([0.0, -0.1, 0.0]),
                np.array([0, 1, 0, 0.8]),
            ),
            (
                "green_three",
                cs.GREEN_CUBE_POSITION + np.array([0.0, -0.2, 0.0]),
                np.array([0, 1, 0, 0.8]),
            ),
            ("blue_one", cs.BLUE_CUBE_POSITION, np.array([0, 1, 1, 0.8])),
            (
                "blue_two",
                cs.BLUE_CUBE_POSITION + np.array([0.0, -0.1, 0.0]),
                np.array([0, 1, 1, 0.8]),
            ),
            (
                "blue_three",
                cs.BLUE_CUBE_POSITION + np.array([0.0, -0.2, 0.0]),
                np.array([0, 1, 1, 0.8]),
            ),
        ]
