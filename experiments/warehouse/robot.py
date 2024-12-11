from typing import Optional

import numpy as np
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet


class PackingPanda(Panda):
    """Panda robot for packing customer orders."""

    def __init__(
        self,
        sim: PyBullet,
        block_gripper: bool = False,
        base_position: Optional[np.ndarray] = None,
        control_type: str = "ee",
    ) -> None:
        """Initialise the packing panda robot."""
        super().__init__(sim, block_gripper, base_position, control_type)
        self.joint_forces = np.array(
            [87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0]
        )

    def set_action(self, action: np.ndarray) -> None:
        """Set the action for the packing panda robot."""
        action = action.copy()
        action = np.clip(action, self.action_space.low, self.action_space.high)  # type: ignore
        if self.control_type == "ee":
            ee_displacement = action[:3]
            target_arm_angles = self.ee_displacement_to_target_arm_angles(
                ee_displacement
            )
        else:
            arm_joint_ctrl = action[:7]
            target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(
                arm_joint_ctrl,
            )

        if self.block_gripper:
            target_fingers_width = 0
        else:
            # Prevent gripper from deforming product meshes
            fingers_ctrl = action[-1]
            target_fingers_width = 0.05 if fingers_ctrl < 0 else 1.0

        target_angles = np.concatenate(
            (
                target_arm_angles,
                [target_fingers_width / 2, target_fingers_width / 2],
            )
        )
        self.control_joints(target_angles=target_angles)
