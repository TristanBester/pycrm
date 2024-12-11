import numpy as np
from panda_gym.envs.core import Task

import experiments.warehouse.constants.environment as ce
import experiments.warehouse.constants.simulation as cs
from experiments.warehouse.config.scene import (
    get_cube_configs,
    get_region_configs,
    get_waypoint_configs,
)


class PackCustomerOrder(Task):
    """Pack a customer order consisting of red, green and blue products."""

    def __init__(
        self,
        sim,
        scene: str = "basic",
        show_waypoints: bool = False,
        show_regions: bool = False,
    ) -> None:
        """Initialise the order packing task."""
        super().__init__(sim=sim)
        self.scene = scene
        self.show_waypoints = show_waypoints
        self.show_regions = show_regions

        with self.sim.no_rendering():
            self._initialise_scene()

    def reset(self) -> None:
        """Reset the task and sample a new goal."""
        self.goal = np.array([])
        self._reset_cubes()

    def get_obs(self) -> np.ndarray:
        """Return a observation of the environment state."""
        return np.array([])

    def get_achieved_goal(self) -> np.ndarray:
        """Return the current position of objects of interest."""
        return np.array([])

    def get_goal(self) -> np.ndarray:
        """Return the current goal positions for objects of interest."""
        return self.goal

    def is_success(self, achieved_goal, desired_goal, info=None) -> np.ndarray:
        """Return True if the task has been solved."""
        return np.array([False])

    def compute_reward(self, achieved_goal, desired_goal, info=None) -> np.ndarray:
        """Return the reward based on the goal and the current state."""
        return np.array([-1 * np.linalg.norm(achieved_goal - desired_goal)])

    def _initialise_scene(self):
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
        for color, position, rgba_color in get_cube_configs():
            self.sim.create_box(
                body_name=f"{color}_cube",
                half_extents=cs.CUBE_HALF_EXTENTS,
                mass=1.0,
                position=position,
                rgba_color=rgba_color,
            )

    def _setup_waypoints(self) -> None:
        """Setup the waypoints."""
        for prefix, red_pos, green_pos, blue_pos in get_waypoint_configs():
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
        for region_type, half_extents, positions, color in get_region_configs():
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

    def _reset_cubes(self) -> None:
        """Reset the cubes to their initial positions."""
        cube_positions = {color: position for color, position, _ in get_cube_configs()}

        for color, position in cube_positions.items():
            self.sim.set_base_pose(
                f"{color}_cube",
                position,
                np.array([0, 0, 0, 1]),
            )
