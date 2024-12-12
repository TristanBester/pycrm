import numpy as np
from panda_gym.pybullet import PyBullet

import experiments.warehouse.constants.environment as ce
import experiments.warehouse.constants.regions as cr
import experiments.warehouse.constants.simulation as cs
import experiments.warehouse.constants.waypoints as cw
from experiments.warehouse.scenes.interface import SceneManager


class BasicSceneManager(SceneManager):
    """Construct a basic scene."""

    def __init__(
        self,
        sim: PyBullet,
        show_ee_identifier: bool = False,
        show_waypoints: bool = False,
        show_regions: bool = False,
        frame_delay: float = 0.01,
    ) -> None:
        """Initialise the scene constructor."""
        self.sim = sim
        self.show_ee_identifier = show_ee_identifier
        self.show_waypoints = show_waypoints
        self.show_regions = show_regions
        self._frame_delay = frame_delay

    @property
    def frame_delay(self) -> float:
        """Get the frame delay for animations in seconds."""
        return self._frame_delay

    def construct(self) -> None:
        """Construct the scene."""
        self._setup_environment()
        self._setup_cubes()

        if self.show_ee_identifier:
            self._setup_ee_pos_identifier()
        if self.show_waypoints:
            self._setup_waypoints()
        if self.show_regions:
            self._setup_regions()

    def update_ee_identifier(self, ee_pos: np.ndarray) -> None:
        """Update the end-effector identifier."""
        if not self.show_ee_identifier:
            return

        self.sim.set_base_pose(
            "ee_pos_identifier",
            ee_pos,
            np.array([0, 0, 0, 1]),
        )

    def animate_red_block(self) -> None:
        """Animate the red block."""
        self.sim.set_base_pose(
            "red_cube",
            cs.RED_CUBE_POSITION,
            np.array([0, 0, 0, 1]),
        )

    def animate_green_block(self) -> None:
        """Animate the green block."""
        self.sim.set_base_pose(
            "green_cube",
            cs.GREEN_CUBE_POSITION,
            np.array([0, 0, 0, 1]),
        )

    def animate_blue_block(self) -> None:
        """Animate the blue block."""
        self.sim.set_base_pose(
            "blue_cube",
            cs.BLUE_CUBE_POSITION,
            np.array([0, 0, 0, 1]),
        )

    def translate_tray(self, destination: str) -> None:
        """Translate the tray to the destination."""
        return

    def _initialise_scene(self) -> None:
        """Initialise the scene."""
        self._setup_environment()
        self._setup_ee_pos_identifier()
        self._setup_cubes()

        if self.show_waypoints:
            self._setup_waypoints()
        if self.show_regions:
            self._setup_regions()

    def _setup_environment(self) -> None:
        """Setup the environment."""
        self.sim.create_plane(z_offset=-0.5)
        self.sim.create_table(length=1.5, height=0.5, width=1.2, x_offset=0.25)

    def _setup_ee_pos_identifier(self) -> None:
        """Setup the end effector position identifier."""
        self.sim.create_sphere(
            body_name="ee_pos_identifier",
            radius=0.01,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=ce.EE_POS_IDENTIFIER_COLOR,
        )

    def _setup_cubes(self) -> None:
        """Setup the cubes."""
        for color, position, rgba_color in self._get_cube_configs():
            self.sim.create_box(
                body_name=f"{color}_cube",
                half_extents=cs.CUBE_HALF_EXTENTS,
                mass=1.0,
                position=position,
                rgba_color=rgba_color,
            )

    def _setup_waypoints(self) -> None:
        """Setup the waypoints."""
        for prefix, red_pos, green_pos, blue_pos in self._get_waypoint_configs():
            for color, position in [
                ("red", red_pos),
                ("green", green_pos),
                ("blue", blue_pos),
            ]:
                self.sim.create_sphere(
                    body_name=f"{prefix}_{color}",
                    radius=0.01,
                    mass=0.0,
                    ghost=True,
                    position=position,
                    rgba_color=ce.WAYPOINT_COLOR,
                )

    def _setup_regions(self) -> None:
        """Setup the regions."""
        for region_type, half_extents, positions, color in self._get_region_configs():
            for cube_color, position in zip(
                ["red", "green", "blue"], positions, strict=True
            ):
                self.sim.create_box(
                    body_name=f"{region_type}_{cube_color}",
                    half_extents=half_extents,
                    mass=0.0,
                    ghost=True,
                    position=position,
                    rgba_color=color,
                )

    def _get_cube_configs(self) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """Return the configuration for the cubes in the scene."""
        return [
            ("red", cs.RED_CUBE_POSITION, np.array([1, 0, 0, 0.8])),
            ("green", cs.GREEN_CUBE_POSITION, np.array([0, 1, 0, 0.8])),
            ("blue", cs.BLUE_CUBE_POSITION, np.array([0, 1, 1, 0.8])),
        ]

    def _get_waypoint_configs(
        self,
    ) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
        """Return the configuration for the waypoints in the scene."""
        return [
            ("grasp", cw.GRASP_RED, cw.GRASP_GREEN, cw.GRASP_BLUE),
            ("above", cw.ABOVE_RED, cw.ABOVE_GREEN, cw.ABOVE_BLUE),
            ("release", cw.RELEASE_RED, cw.RELEASE_GREEN, cw.RELEASE_BLUE),
        ]

    def _get_region_configs(
        self,
    ) -> list[tuple[str, np.ndarray, list[np.ndarray], np.ndarray]]:
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
                ce.SAFE_REGION_COLOR,
            ),
            (
                "tight_region",
                cr.TIGHT_REGION_HALF_EXTENTS,
                [
                    cr.TIGHT_REGION_RED_COM,
                    cr.TIGHT_REGION_GREEN_COM,
                    cr.TIGHT_REGION_BLUE_COM,
                ],
                ce.TIGHT_REGION_COLOR,
            ),
            (
                "release_region",
                cr.RELEASE_REGION_HALF_EXTENTS,
                [
                    cr.RELEASE_REGION_RED_COM,
                    cr.RELEASE_REGION_GREEN_COM,
                    cr.RELEASE_REGION_BLUE_COM,
                ],
                ce.RELEASE_REGION_COLOR,
            ),
        ]
