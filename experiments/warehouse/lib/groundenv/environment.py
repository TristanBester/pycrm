from typing import Optional

import numpy as np
from panda_gym.envs.core import RobotTaskEnv
from panda_gym.pybullet import PyBullet

import experiments.warehouse.lib.groundenv.constants.environment as ce
from experiments.warehouse.lib.groundenv.robot import PackingPanda
from experiments.warehouse.lib.groundenv.task import PackCustomerOrder


class PackCustomerOrderEnvironment(RobotTaskEnv):
    """Order packing warehouse environment."""

    def __init__(
        self,
        render_mode: str = "rgb_array",
        control_type: str = "ee",
        renderer: str = "Tiny",
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 135,
        render_pitch: float = -20,
        render_roll: float = 0,
        scene: str = "basic",
        show_ee_identifier: bool = False,
        show_waypoints: bool = False,
        show_regions: bool = False,
    ) -> None:
        """Initialise the order packing warehouse environment."""
        # Initialise the simulation
        sim = PyBullet(
            render_mode=render_mode,
            renderer=renderer,
            background_color=ce.BACKGROUND_COLOR,
        )

        # Initialise the robot
        robot = PackingPanda(
            sim=sim,
            block_gripper=False,
            base_position=ce.ROBOT_BASE_POSITION,
            control_type=control_type,
        )

        # Initialise the task
        task = PackCustomerOrder(
            sim=sim,
            scene=scene,
            show_waypoints=show_waypoints,
            show_regions=show_regions,
            show_ee_identifier=show_ee_identifier,
        )

        # Initialise the environment
        super().__init__(
            robot=robot,
            task=task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=render_target_position,
            render_distance=render_distance,
            render_yaw=render_yaw,
            render_pitch=render_pitch,
            render_roll=render_roll,
        )

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        """Step the environment."""
        action = self._preprocess_action(action)
        return super().step(action)

    def _preprocess_action(self, action: np.ndarray) -> np.ndarray:
        """Preprocess the action."""
        # Clip end-effector displacement
        action[:-1] = np.clip(action[:-1], -0.1, 0.1)

        # Set gripper action
        if action[-1] <= 0:
            # Close gripper
            action[-1] = -1.0
        else:
            # Open gripper
            action[-1] = 1.0
        return action
