import numpy as np
from panda_gym.envs.core import Task

import experiments.warehouse_exp.lib.constants.simulation as cs
import experiments.warehouse_exp.lib.constants.waypoints as cw


class PickUp(Task):
    def __init__(self, sim, show_regions):
        super().__init__(sim)
        self.show_regions = show_regions

        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self):
        """Setup the objects in the scene."""
        self.sim.create_plane(z_offset=-0.5)
        self.sim.create_table(length=1.5, height=0.5, width=1.2, x_offset=0.25)

        # Sphere used to visualise end effector position
        self.sim.create_sphere(
            body_name="ee_pos_identifier",
            radius=0.01,
            mass=0.0,
            ghost=True,
            position=np.array([0.5, 0.25, 0.1]),
            rgba_color=np.array([1, 0, 0, 0.3]),
        )

        # Cubes
        self.sim.create_box(
            body_name="green_cube",
            half_extents=cs.CUBE_HALF_EXTENTS,
            mass=1.0,
            position=cs.GREEN_CUBE_POSITION,
            rgba_color=np.array([0, 1, 0, 0.8]),
        )
        self.sim.create_box(
            body_name="red_cube",
            half_extents=cs.CUBE_HALF_EXTENTS,
            mass=1.0,
            position=cs.RED_CUBE_POSITION,
            rgba_color=np.array([1, 0, 0, 0.8]),
        )
        self.sim.create_box(
            body_name="blue_cube",
            half_extents=cs.CUBE_HALF_EXTENTS,
            mass=1.0,
            position=cs.BLUE_CUBE_POSITION,
            rgba_color=np.array([0, 1, 1, 0.8]),
        )

        # Above waypoints green
        self.sim.create_sphere(
            body_name="above_green",
            radius=0.01,
            mass=0.0,
            ghost=True,
            position=cw.ABOVE_GREEN,
            rgba_color=np.array([0, 0, 1, 0.3]),
        )
        self.sim.create_sphere(
            body_name="above_red",
            radius=0.01,
            mass=0.0,
            ghost=True,
            position=cw.ABOVE_RED,
            rgba_color=np.array([0, 0, 1, 0.3]),
        )
        # self.sim.create_sphere(
        #     body_name="above_blue",
        #     radius=0.01,
        #     mass=0.0,
        #     ghost=True,
        #     position=cw.ABOVE_BLUE,
        #     rgba_color=np.array([0, 0, 1, 0.3]),
        # )

        # Grasp waypoints green
        self.sim.create_sphere(
            body_name="grasp_green",
            radius=0.01,
            mass=0.0,
            ghost=True,
            position=cw.GRASP_GREEN,
            rgba_color=np.array([0, 0, 1, 0.3]),
        )
        self.sim.create_sphere(
            body_name="grasp_red",
            radius=0.01,
            mass=0.0,
            ghost=True,
            position=cw.GRASP_RED,
            rgba_color=np.array([0, 0, 1, 0.3]),
        )
        # self.sim.create_sphere(
        #     body_name="grasp_blue",
        #     radius=0.01,
        #     mass=0.0,
        #     ghost=True,
        #     position=cw.GRASP_BLUE,
        #     rgba_color=np.array([0, 0, 1, 0.3]),
        # )

        # Drop waypoints green
        self.sim.create_sphere(
            body_name="release_green",
            radius=0.01,
            mass=0.0,
            ghost=True,
            position=cw.RELEASE_GREEN,
            rgba_color=np.array([0, 0, 1, 0.3]),
        )
        self.sim.create_sphere(
            body_name="release_red",
            radius=0.01,
            mass=0.0,
            ghost=True,
            position=cw.RELEASE_RED,
            rgba_color=np.array([0, 0, 1, 0.3]),
        )
        # self.sim.create_sphere(
        #     body_name="release_blue",
        #     radius=0.01,
        #     mass=0.0,
        #     ghost=True,
        #     position=cw.RELEASE_BLUE,
        #     rgba_color=np.array([0, 0, 1, 0.3]),
        # )
        #
        # if self.show_regions:
        #     # Safe regions green
        #     self.sim.create_box(
        #         body_name="safe_region_green",
        #         half_extents=cr.SAFE_REGION_HALF_EXTENTS,
        #         mass=0.0,
        #         ghost=True,
        #         position=cr.SAFE_REGION_GREEN_COM,
        #         rgba_color=np.array([1, 0, 1, 0.2]),
        #     )
        #     self.sim.create_box(
        #         body_name="safe_region_red",
        #         half_extents=cr.SAFE_REGION_HALF_EXTENTS,
        #         mass=0.0,
        #         ghost=True,
        #         position=cr.SAFE_REGION_RED_COM,
        #         rgba_color=np.array([1, 0, 1, 0.2]),
        #     )
        #     self.sim.create_box(
        #         body_name="safe_region_blue",
        #         half_extents=cr.SAFE_REGION_HALF_EXTENTS,
        #         mass=0.0,
        #         ghost=True,
        #         position=cr.SAFE_REGION_BLUE_COM,
        #         rgba_color=np.array([1, 0, 1, 0.2]),
        #     )
        #
        #     # Tight regions green
        #     self.sim.create_box(
        #         body_name="tight_region_green",
        #         half_extents=cr.TIGHT_REGION_HALF_EXTENTS,
        #         mass=0.0,
        #         ghost=True,
        #         position=cr.TIGHT_REGION_GREEN_COM,
        #         rgba_color=np.array([1, 1, 0, 0.2]),
        #     )
        #     self.sim.create_box(
        #         body_name="tight_region_red",
        #         half_extents=cr.TIGHT_REGION_HALF_EXTENTS,
        #         mass=0.0,
        #         ghost=True,
        #         position=cr.TIGHT_REGION_RED_COM,
        #         rgba_color=np.array([1, 1, 0, 0.2]),
        #     )
        #     self.sim.create_box(
        #         body_name="tight_region_blue",
        #         half_extents=cr.TIGHT_REGION_HALF_EXTENTS,
        #         mass=0.0,
        #         ghost=True,
        #         position=cr.TIGHT_REGION_BLUE_COM,
        #         rgba_color=np.array([1, 1, 0, 0.2]),
        #     )
        #
        #     # Release regions
        #     self.sim.create_box(
        #         body_name="release_region_green",
        #         half_extents=cr.RELEASE_REGION_HALF_EXTENTS,
        #         mass=0.0,
        #         ghost=True,
        #         position=cr.RELEASE_REGION_GREEN_COM,
        #         rgba_color=np.array([1, 0, 1, 0.2]),
        #     )
        #     self.sim.create_box(
        #         body_name="release_region_red",
        #         half_extents=cr.RELEASE_REGION_HALF_EXTENTS,
        #         mass=0.0,
        #         ghost=True,
        #         position=cr.RELEASE_REGION_RED_COM,
        #         rgba_color=np.array([1, 0, 1, 0.2]),
        #     )
        #     self.sim.create_box(
        #         body_name="release_region_blue",
        #         half_extents=cr.RELEASE_REGION_HALF_EXTENTS,
        #         mass=0.0,
        #         ghost=True,
        #         position=cr.RELEASE_REGION_BLUE_COM,
        #         rgba_color=np.array([1, 0, 1, 0.2]),
        #     )

    def reset(self) -> None:
        """Reset the task and sample a new goal."""
        self.goal = self._sample_goal()
        self.reset_green_block_pose()
        self.reset_red_block_pose()

    def reset_green_block_pose(self):
        self.sim.set_base_pose(
            "green_cube",
            cs.GREEN_CUBE_POSITION,
            np.array([0, 0, 0, 1]),
        )

    def reset_red_block_pose(self):
        self.sim.set_base_pose(
            "red_cube",
            cs.RED_CUBE_POSITION,
            np.array([0, 0, 0, 1]),
        )

    def reset_blue_block_pose(self):
        self.sim.set_base_pose(
            "blue_cube",
            cs.BLUE_CUBE_POSITION,
            np.array([0, 0, 0, 1]),
        )

    def update_waypoints(self, ee_pos: np.ndarray):
        self.sim.set_base_pose(
            "ee_pos_identifier",
            ee_pos + np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
        )

    def get_obs(self) -> np.ndarray:
        """Return a observation of the environment state."""
        return np.array([])

    def get_achieved_goal(self) -> np.ndarray:
        """Return the current position of objects of interest."""
        return np.array([])

    def get_goal(self) -> np.ndarray:
        """Return the current goal positions for objects of interest."""
        return self.goal

    def is_success(self, achieved_goal, desired_goal, info={}) -> np.ndarray:
        """Return True if the task has been solved."""
        # The task is considered successful if the cube is close to the target.
        del info, desired_goal, achieved_goal
        return np.array([False])

    def compute_reward(self, achieved_goal, desired_goal, info={}) -> np.ndarray:
        """Return the reward based on the goal and the current state."""
        del info
        return np.array([-1 * np.linalg.norm(achieved_goal - desired_goal)])

    def _sample_goal(self) -> np.ndarray:
        """Sample a new goal for the task."""
        return np.array([])
