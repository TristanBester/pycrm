import numpy as np

from experiments.warehouse.lib.machines.reward.reward import RewardFunction


def create_constant_reward(
    constant: float, enable_rescaling: bool = False
) -> RewardFunction:
    """Create a constant reward function."""

    def f_constant(obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray) -> float:
        return constant

    # Return reward function in the range {constant}
    return RewardFunction(
        reward_fn=f_constant,
        min_reward=constant,
        max_reward=constant,
        enable_rescaling=enable_rescaling,
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

    # Return reward function in the range [0, 1]
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

    # Return reward function in the range [0, 1]
    return RewardFunction(
        reward_fn=f_penalty_waypoint,
        min_reward=-max_distance + penalty,
        max_reward=penalty,
        enable_rescaling=True,
    )


def recaled_reward_function(
    rf: RewardFunction,
    r_min: float,
    r_max: float,
) -> RewardFunction:
    """Recale a reward function to a new range."""
    return RewardFunction(
        reward_fn=rf._reward_fn,
        min_reward=r_min,
        max_reward=r_max,
        enable_rescaling=True,
    )
