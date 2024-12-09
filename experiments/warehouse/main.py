import numpy as np

from experiments.warehouse.environment import PackCustomerOrderEnvironment
from experiments.warehouse.wrapper import EEStateWrapper

if __name__ == "__main__":
    import time

    env = PackCustomerOrderEnvironment(render_mode="human", control_type="ee")
    env = EEStateWrapper(env)

    obs, _ = env.reset()
    for _ in range(1000):
        curr_pos = obs[:3]

        # desired_pos = CUBE_ONE_POSITION
        desired_pos = np.array([0.5, 0.15, 0.5])
        action = np.clip((desired_pos - curr_pos), -1, 1)
        action = np.concatenate([action, [1]])
        obs, reward, terminated, truncated, info = env.step(action)
        # env.task.update_waypoints(obs["observation"][:3])
        env.render()

        time.sleep(0.1)
