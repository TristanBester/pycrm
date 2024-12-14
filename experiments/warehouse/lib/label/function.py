import numpy as np

import experiments.warehouse.lib.groundenv.constants.regions as cr
import experiments.warehouse.lib.groundenv.constants.waypoints as cw
from crm.label import LabellingFunction
from experiments.warehouse.lib.label.events import WarehouseEvent


class WarehouseLabellingFunction(LabellingFunction[np.ndarray, np.ndarray]):
    """Labelling function for the warehouse environment."""

    def __init__(
        self,
        grasp_waypoint_xy_tol=0.025,
        grasp_waypoint_z_tol=0.01,
        position_waypoint_tol=0.025,
        vel_tol=0.01,
    ) -> None:
        """Initialise the labelling function."""
        super().__init__()
        self.grasp_waypoint_xy_tol = grasp_waypoint_xy_tol
        self.grasp_waypoint_z_tol = grasp_waypoint_z_tol
        self.position_waypoint_tol = position_waypoint_tol
        self.vel_tol = vel_tol

    @LabellingFunction.event
    def above_red(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> WarehouseEvent | None:
        """Return if the end-effector is above the red cube."""
        if self._test_ee_near_waypoint(next_obs[:3], cw.ABOVE_RED):
            return WarehouseEvent.ABOVE_RED
        return None

    @LabellingFunction.event
    def above_green(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> WarehouseEvent | None:
        """Return if the end-effector is above the green cube."""
        if self._test_ee_near_waypoint(next_obs[:3], cw.ABOVE_GREEN):
            return WarehouseEvent.ABOVE_GREEN
        return None

    @LabellingFunction.event
    def above_blue(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> WarehouseEvent | None:
        """Return if the end-effector is above the blue cube."""
        if self._test_ee_near_waypoint(next_obs[:3], cw.ABOVE_BLUE):
            return WarehouseEvent.ABOVE_BLUE
        return None

    @LabellingFunction.event
    def grasp_red(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> WarehouseEvent | None:
        """Return if the end-effector is grasping the red cube."""
        if self._test_ee_near_waypoint(next_obs[:3], cw.GRASP_RED):
            return WarehouseEvent.GRASP_RED
        return None

    @LabellingFunction.event
    def grasp_green(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> WarehouseEvent | None:
        """Return if the end-effector is grasping the green cube."""
        if self._test_ee_near_waypoint(next_obs[:3], cw.GRASP_GREEN):
            return WarehouseEvent.GRASP_GREEN
        return None

    @LabellingFunction.event
    def grasp_blue(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> WarehouseEvent | None:
        """Return if the end-effector is grasping the blue cube."""
        if self._test_ee_near_waypoint(next_obs[:3], cw.GRASP_BLUE):
            return WarehouseEvent.GRASP_BLUE
        return None

    @LabellingFunction.event
    def release_red(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> WarehouseEvent | None:
        """Return if the end-effector is releasing the red cube."""
        if self._test_ee_near_waypoint(next_obs[:3], cw.RELEASE_RED):
            return WarehouseEvent.RELEASE_RED
        return None

    @LabellingFunction.event
    def release_green(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> WarehouseEvent | None:
        """Return if the end-effector is releasing the green cube."""
        if self._test_ee_near_waypoint(next_obs[:3], cw.RELEASE_GREEN):
            return WarehouseEvent.RELEASE_GREEN
        return None

    @LabellingFunction.event
    def release_blue(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> WarehouseEvent | None:
        """Return if the end-effector is releasing the blue cube."""
        if self._test_ee_near_waypoint(next_obs[:3], cw.RELEASE_BLUE):
            return WarehouseEvent.RELEASE_BLUE
        return None

    @LabellingFunction.event
    def safe_region_red(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> WarehouseEvent | None:
        """Return if the end-effector is in the safe region of the red cube."""
        if self._test_ee_within_region(
            next_obs[:3], cr.SAFE_REGION_RED_COM, cr.SAFE_REGION_HALF_EXTENTS
        ):
            return WarehouseEvent.SAFE_REGION_RED
        return None

    @LabellingFunction.event
    def safe_region_green(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> WarehouseEvent | None:
        """Return if the end-effector is in the safe region of the green cube."""
        if self._test_ee_within_region(
            next_obs[:3], cr.SAFE_REGION_GREEN_COM, cr.SAFE_REGION_HALF_EXTENTS
        ):
            return WarehouseEvent.SAFE_REGION_GREEN
        return None

    @LabellingFunction.event
    def safe_region_blue(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> WarehouseEvent | None:
        """Return if the end-effector is in the safe region of the blue cube."""
        if self._test_ee_within_region(
            next_obs[:3], cr.SAFE_REGION_BLUE_COM, cr.SAFE_REGION_HALF_EXTENTS
        ):
            return WarehouseEvent.SAFE_REGION_BLUE
        return None

    @LabellingFunction.event
    def tight_region_red(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> WarehouseEvent | None:
        """Return if the end-effector is in the tight region of the red cube."""
        if self._test_ee_within_region(
            next_obs[:3], cr.TIGHT_REGION_RED_COM, cr.TIGHT_REGION_HALF_EXTENTS
        ):
            return WarehouseEvent.TIGHT_REGION_RED
        return None

    @LabellingFunction.event
    def tight_region_green(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> WarehouseEvent | None:
        """Return if the end-effector is in the tight region of the green cube."""
        if self._test_ee_within_region(
            next_obs[:3], cr.TIGHT_REGION_GREEN_COM, cr.TIGHT_REGION_HALF_EXTENTS
        ):
            return WarehouseEvent.TIGHT_REGION_GREEN
        return None

    @LabellingFunction.event
    def tight_region_blue(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> WarehouseEvent | None:
        """Return if the end-effector is in the tight region of the blue cube."""
        if self._test_ee_within_region(
            next_obs[:3], cr.TIGHT_REGION_BLUE_COM, cr.TIGHT_REGION_HALF_EXTENTS
        ):
            return WarehouseEvent.TIGHT_REGION_BLUE
        return None

    @LabellingFunction.event
    def release_region_red(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> WarehouseEvent | None:
        """Return if the end-effector is in the release region of the red cube."""
        if self._test_ee_within_region(
            next_obs[:3], cr.RELEASE_REGION_RED_COM, cr.RELEASE_REGION_HALF_EXTENTS
        ):
            return WarehouseEvent.RELEASE_REGION_RED
        return None

    @LabellingFunction.event
    def release_region_green(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> WarehouseEvent | None:
        """Return if the end-effector is in the release region of the green cube."""
        if self._test_ee_within_region(
            next_obs[:3], cr.RELEASE_REGION_GREEN_COM, cr.RELEASE_REGION_HALF_EXTENTS
        ):
            return WarehouseEvent.RELEASE_REGION_GREEN
        return None

    @LabellingFunction.event
    def release_region_blue(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> WarehouseEvent | None:
        """Return if the end-effector is in the release region of the blue cube."""
        if self._test_ee_within_region(
            next_obs[:3], cr.RELEASE_REGION_BLUE_COM, cr.RELEASE_REGION_HALF_EXTENTS
        ):
            return WarehouseEvent.RELEASE_REGION_BLUE
        return None

    @LabellingFunction.event
    def velocity_low(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> WarehouseEvent | None:
        """Return if the end-effector velocity is low."""
        ee_vel = next_obs[3:6]
        if np.linalg.norm(ee_vel, ord=np.inf) < self.vel_tol:
            return WarehouseEvent.VELOCITY_LOW
        return None

    @LabellingFunction.event
    def gripper_open_action_executed(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> WarehouseEvent | None:
        """Return if the gripper open action was executed."""
        if action[-1] > 0.0:
            return WarehouseEvent.GRIPPER_OPEN_ACTION_EXECUTED
        return None

    @LabellingFunction.event
    def gripper_closed(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> WarehouseEvent | None:
        """Return if the gripper is closed."""
        gripper_state = next_obs[6]
        if np.abs(gripper_state) < 0.065:
            return WarehouseEvent.GRIPPER_CLOSED
        return None

    @LabellingFunction.event
    def gripper_open(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> WarehouseEvent | None:
        """Return if the gripper is open."""
        gripper_state = next_obs[6]
        if np.abs(gripper_state) > 0.07:
            return WarehouseEvent.GRIPPER_OPEN
        return None

    def _test_ee_near_waypoint(
        self, ee_pos: np.ndarray, waypoint_pos: np.ndarray
    ) -> bool:
        """Test if the end-effector is near the waypoint."""
        ee_pos_x, ee_pos_y, ee_pos_z = ee_pos
        waypoint_pos_x, waypoint_pos_y, waypoint_pos_z = waypoint_pos[:3]

        # x_test = np.abs(ee_pos_x - waypoint_pos_x) <= self.position_waypoint_tol
        # y_test = np.abs(ee_pos_y - waypoint_pos_y) <= self.position_waypoint_tol
        # z_test = np.abs(ee_pos_z - waypoint_pos_z) <= self.position_waypoint_tol

        # print(
        #     "X TEST",
        #     x_test,
        #     np.abs(ee_pos_x - waypoint_pos_x),
        #     self.position_waypoint_tol,
        # )
        # print(
        #     "Y TEST",
        #     y_test,
        #     np.abs(ee_pos_y - waypoint_pos_y),
        #     self.position_waypoint_tol,
        # )
        # print(
        #     "Z TEST",
        #     z_test,
        #     np.abs(ee_pos_z - waypoint_pos_z),
        #     self.position_waypoint_tol,
        # )

        return (
            np.abs(ee_pos_x - waypoint_pos_x) <= self.position_waypoint_tol
            and np.abs(ee_pos_y - waypoint_pos_y) <= self.position_waypoint_tol
            and np.abs(ee_pos_z - waypoint_pos_z) <= self.position_waypoint_tol
        )

    def _test_ee_within_region(
        self,
        ee_pos: np.ndarray,
        region_com: np.ndarray,
        region_half_extents: np.ndarray,
    ) -> bool:
        """Test if the end-effector is within the region."""
        x_test = (
            region_com[0] - region_half_extents[0]
            < ee_pos[0]
            < region_com[0] + region_half_extents[0]
        )
        y_test = (
            region_com[1] - region_half_extents[1]
            < ee_pos[1]
            < region_com[1] + region_half_extents[1]
        )
        z_test = (
            region_com[2] - region_half_extents[2]
            < ee_pos[2]
            < region_com[2] + region_half_extents[2]
        )
        return x_test and y_test and z_test
