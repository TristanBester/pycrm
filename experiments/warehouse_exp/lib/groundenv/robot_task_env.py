from typing import Optional

import numpy as np
from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet

from experiments.warehouse_exp.lib.groundenv.task import PickUp


class ModdedPanda(Panda):
    def __init__(
        self,
        sim: PyBullet,
        block_gripper: bool = False,
        base_position: Optional[np.ndarray] = None,
        control_type: str = "ee",
    ) -> None:
        super().__init__(sim, block_gripper, base_position, control_type)
        self.joint_forces = np.array(
            [87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0]
        )

    def set_action(self, action: np.ndarray) -> None:
        action = action.copy()  # ensure action don't change
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.control_type == "ee":
            ee_displacement = action[:3]
            target_arm_angles = self.ee_displacement_to_target_arm_angles(
                ee_displacement
            )
        else:
            arm_joint_ctrl = action[:7]
            target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)

        if self.block_gripper:
            target_fingers_width = 0
        else:
            # fingers_ctrl = action[-1] * 0.2  # limit maximum change in position
            fingers_ctrl = action[-1]
            # fingers_width = self.get_fingers_width()
            # target_fingers_width = fingers_width + fingers_ctrl

            if fingers_ctrl < 0:
                # the commented out value works for the handwritten policy, not for the trained one (which is more aggressive)
                # target_fingers_width = 0.0514
                target_fingers_width = 0.05
            else:
                target_fingers_width = 1.0

        target_angles = np.concatenate(
            (target_arm_angles, [target_fingers_width / 2, target_fingers_width / 2])
        )
        self.control_joints(target_angles=target_angles)


class PickUpEnv(RobotTaskEnv):
    def __init__(
        self,
        render_mode: str = "rgb_array",
        control_type: str = "ee",
        renderer: str = "Tiny",
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 90,
        render_pitch: float = -15,
        render_roll: float = 0,
    ) -> None:
        sim = PyBullet(
            render_mode=render_mode,
            renderer=renderer,
            background_color=np.array([128, 128, 128]),
        )
        robot = ModdedPanda(
            sim,
            block_gripper=False,
            base_position=np.array([-0.1, 0.0, 0.0]),
            control_type=control_type,
        )

        if render_mode == "rgb_array":
            show_regions = False
        else:
            show_regions = True
        task = PickUp(sim, show_regions)
        super().__init__(
            robot,
            task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=render_target_position,
            render_distance=render_distance,
            render_yaw=render_yaw,
            render_pitch=render_pitch,
            render_roll=render_roll,
        )


# FIXME: replace this with an actual gym.ObservationWrapper subclass
class WrappedPickPlaceEnv:
    def __init__(self, render_mode: str = "human"):
        self._env = PickUpEnv(render_mode=render_mode)

    def reset(self):
        obs, _ = self._env.reset()
        return obs["observation"][:7], _

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        return obs["observation"][:7], reward, terminated, truncated, info


if __name__ == "__main__":
    import time

    env = PickUpEnv(render_mode="human")
    obs, _ = env.reset()

    print(np.round(obs["observation"][:3], 3))

    print(env.action_space)

    for _ in range(1000):
        curr_pos = obs["observation"][:3]

        # desired_pos = CUBE_ONE_POSITION
        desired_pos = np.array([0.5, 0.15, 0.5])
        action = np.clip((desired_pos - curr_pos), -1, 1)
        action = np.concatenate([action, [1]])
        obs, reward, terminated, truncated, info = env.step(action)
        env.task.update_waypoints(obs["observation"][:3])
        env.render()

        time.sleep(0.1)
