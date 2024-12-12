import numpy as np

from experiments.warehouse.lib.machine.reward.reward import RewardFunction


def create_constant_reward(constant: float) -> RewardFunction:
    """Create a constant reward function."""

    def f_constant(obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray) -> float:
        return constant

    return RewardFunction(
        reward_fn=f_constant,
        min_reward=constant,
        max_reward=constant,
        enable_rescaling=False,
    )


def create_waypoint_reward(waypoint: np.ndarray, max_distance: float) -> RewardFunction:
    """Create a reward function that rewards distance to a target point."""

    def f_waypoint(
        obs: np.ndarray,
        action: np.ndarray,
        obs_next: np.ndarray,
    ) -> float:
        ee_pos_next = obs_next[:3]
        pos_err_next = float(np.linalg.norm(ee_pos_next - waypoint))
        return -pos_err_next

    return RewardFunction(
        reward_fn=f_waypoint,
        min_reward=-max_distance,
        max_reward=0.0,
        enable_rescaling=True,
    )


def create_penalty_waypoint_reward(
    waypoint: np.ndarray, penalty: float, max_distance: float
) -> RewardFunction:
    """Create a reward function that penalizes distance to a target point."""

    def f_penalty_waypoint(
        obs: np.ndarray,
        action: np.ndarray,
        obs_next: np.ndarray,
    ) -> float:
        ee_pos_next = obs_next[:3]
        pos_err_next = float(np.linalg.norm(ee_pos_next - waypoint))
        return -pos_err_next + penalty

    return RewardFunction(
        reward_fn=f_penalty_waypoint,
        min_reward=-max_distance + penalty,
        max_reward=penalty,
        enable_rescaling=True,
    )
