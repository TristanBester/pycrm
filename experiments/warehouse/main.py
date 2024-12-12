import time

import numpy as np

from experiments.warehouse.environment import PackCustomerOrderEnvironment
from experiments.warehouse.wrapper import EEStateWrapper

if __name__ == "__main__":
    env = PackCustomerOrderEnvironment(
        render_mode="human", control_type="ee", scene="fancy"
    )
    env = EEStateWrapper(env)

    obs, _ = env.reset()
    for step in range(1000):
        if step == 50:
            env.sim.set_base_pose(
                "red_one_cube",
                np.array([0.5, 0.35, 0.5]),
                np.array([0, 0, 0, 1]),
            )
        elif step == 100:
            env.task.scene_manager.animate_red_block()
        elif step == 150:
            env.sim.set_base_pose(
                "red_two_cube",
                np.array([0.5, 0.35, 0.5]),
                np.array([0, 0, 0, 1]),
            )
        elif step == 200:
            env.task.scene_manager.animate_red_block()
        elif step == 250:
            env.sim.set_base_pose(
                "green_one_cube",
                np.array([0.5, 0.35, 0.5]),
                np.array([0, 0, 0, 1]),
            )
            env.task.scene_manager.translate_tray(destination="green")
        elif step == 300:
            env.task.scene_manager.animate_green_block()
        elif step == 350:
            env.sim.set_base_pose(
                "green_two_cube",
                np.array([0.5, 0.35, 0.5]),
                np.array([0, 0, 0, 1]),
            )
        elif step == 400:
            env.task.scene_manager.animate_green_block()
        elif step == 450:
            env.sim.set_base_pose(
                "blue_one_cube",
                np.array([0.5, 0.35, 0.5]),
                np.array([0, 0, 0, 1]),
            )
        elif step == 500:
            env.task.scene_manager.animate_blue_block()
            env.task.scene_manager.translate_tray(destination="blue")
        elif step == 550:
            env.sim.set_base_pose(
                "blue_two_cube",
                np.array([0.5, 0.35, 0.5]),
                np.array([0, 0, 0, 1]),
            )
        elif step == 600:
            env.task.scene_manager.animate_blue_block()
        elif step == 650:
            env.task.scene_manager.translate_tray(destination="end")

        curr_pos = obs[:3]
        curr_pos = obs[:3]

        # desired_pos = CUBE_ONE_POSITION
        desired_pos = np.array([0.5, 0.15, 0.5])
        action = np.clip((desired_pos - curr_pos), -1, 1)
        action = np.concatenate([action, [1]])
        obs, reward, terminated, truncated, info = env.step(action)
        # env.task.update_waypoints(obs["observation"][:3])
        env.render()

        time.sleep(0.1)
