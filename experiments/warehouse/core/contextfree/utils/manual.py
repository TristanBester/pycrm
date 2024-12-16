import time
from warnings import filterwarnings

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from experiments.warehouse.lib.groundenv.constants import waypoints as cw

filterwarnings("ignore")


def resolve_current_waypoint(u: int) -> np.ndarray:
    """Resolve the current waypoint."""
    match u:
        case 0:
            return np.array([0.0, 0.0, 0.1])
        case 10:
            return cw.ABOVE_RED
        case 9 | 8:
            return cw.GRASP_RED
        case 7 | 6:
            return cw.RELEASE_RED
        case 5:
            return cw.ABOVE_GREEN
        case 4 | 3:
            return cw.GRASP_GREEN
        case 2 | 1:
            return cw.RELEASE_GREEN
        case _:
            raise ValueError(f"Invalid state {u}")


def resolve_gripper_state(u: int) -> float:
    """Resolve the gripper state."""
    match u:
        case 0 | 2 | 3 | 7 | 8:
            return -1.0
        case 4 | 5 | 1 | 10 | 9 | 6:
            return 1.0
        case _:
            raise ValueError(f"Invalid state {u}")


def main() -> None:
    """Solve the task manually."""
    # 1100 steps to solve the task & (-228)
    env = gym.make(
        "Warehouse-ContextFree-v0",
        ground_env_kwargs={"render_mode": "human"},
        crossproduct_kwargs={"max_steps": 10000},
    )
    obs, _ = env.reset()

    returns = []

    for _ in range(10000):
        ee_pos = obs[:3]

        waypoint = resolve_current_waypoint(env.u)  # type: ignore
        delta = np.clip(waypoint - ee_pos, -0.25, 0.25)
        gripper_state = resolve_gripper_state(env.u)  # type: ignore

        action = np.concatenate([delta, np.array([gripper_state])])

        last_u = env.u  # type: ignore
        last_c = env.c  # type: ignore
        obs, reward, terminated, truncated, _ = env.step(action)

        returns.append(reward)

        if last_u != env.u:  # type: ignore
            print(f"State transition: {last_u} -> {env.u}")  # type: ignore
        if last_c != env.c:  # type: ignore
            print(f"Counter transition: {last_c} -> {env.c}")  # type: ignore

        if last_c[0] != env.c[0]:  # type: ignore
            env.ground_env.task.scene_manager.animate_red_block()  # type: ignore
        elif last_c[1] != env.c[1]:  # type: ignore
            env.ground_env.task.scene_manager.animate_green_block()  # type: ignore

        if terminated or truncated:
            print(f"Terminated: {terminated}, truncated: {truncated}")
            break

        env.render()
        time.sleep(0.01)

    plt.plot(returns)
    plt.ylim(-10, 10)
    plt.show()

    print(sum(returns))


if __name__ == "__main__":
    main()
