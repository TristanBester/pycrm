import numpy as np
from panda_gym.envs.core import Task

from experiments.warehouse.scenes import construct_scene


class PackCustomerOrder(Task):
    """Pack a customer order consisting of red, green and blue products."""

    def __init__(
        self,
        sim,
        scene: str = "fancy",
        show_waypoints: bool = False,
        show_regions: bool = False,
    ) -> None:
        """Initialise the order packing task."""
        super().__init__(sim=sim)
        self.scene = scene
        self.show_waypoints = show_waypoints
        self.show_regions = show_regions

        with self.sim.no_rendering():
            construct_scene(self.sim, self.scene)

    def reset(self) -> None:
        """Reset the task and sample a new goal."""
        self.goal = np.array([])

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
