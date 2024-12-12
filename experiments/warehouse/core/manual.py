import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from experiments.warehouse.lib.crossproduct.crossproduct import WarehouseCrossProduct
from experiments.warehouse.lib.groundenv.constants import waypoints as cw
from experiments.warehouse.lib.label.function import WarehouseLabellingFunction
from experiments.warehouse.lib.machine.machine import WarehouseCountingRewardMachine


def create_env() -> WarehouseCrossProduct:
    """Create the warehouse environment."""
    ground_env = gym.make(
        "Warehouse-v0",
        render_mode="human",
    )
    crm = WarehouseCountingRewardMachine()
    lf = WarehouseLabellingFunction()
    env = WarehouseCrossProduct(ground_env, crm, lf)
    return env


def resolve_current_waypoint(u: int) -> np.ndarray:
    """Resolve the current waypoint."""
    match u:
        case 0:
            return np.array([0.0, 0.0, 0.1])
        case 5:
            return cw.ABOVE_RED
        case 4 | 3:
            return cw.GRASP_RED
        case 2 | 1:
            return cw.RELEASE_RED
        case _:
            raise ValueError(f"Invalid state {u}")


def resolve_gripper_state(u: int) -> float:
    """Resolve the gripper state."""
    match u:
        case 0 | 2 | 3:
            return -1.0
        case 4 | 5 | 1:
            return 1.0
        case _:
            raise ValueError(f"Invalid state {u}")


def main() -> None:
    """Solve the task manually."""
    # 175 steps to solve the task & (-228)
    env = create_env()
    obs, _ = env.reset()

    returns = []

    for _ in range(1000):
        ee_pos = obs[:3]
        waypoint = resolve_current_waypoint(env._u)
        delta = np.clip(waypoint - ee_pos, -0.25, 0.25)
        gripper_state = resolve_gripper_state(env._u)

        action = np.concatenate([delta, np.array([gripper_state])])

        last_u = env._u
        last_c = env._c
        obs, reward, terminated, truncated, _ = env.step(action)

        returns.append(reward)

        if last_u != env._u:
            print(f"State transition: {last_u} -> {env._u}")
        if last_c != env._c:
            print(f"Counter transition: {last_c} -> {env._c}")

        if terminated or truncated:
            print(f"Terminated: {terminated}, truncated: {truncated}")
            break

        env.render()
        time.sleep(0.03)

    plt.plot(returns)
    plt.ylim(-6, 10)
    plt.show()

    print(sum(returns))


if __name__ == "__main__":
    main()
