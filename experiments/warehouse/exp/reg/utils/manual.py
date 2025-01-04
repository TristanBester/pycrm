import time
from warnings import filterwarnings

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import experiments.warehouse.lib.groundenv.constants.waypoints as cw

filterwarnings("ignore")

env = gym.make(
    "Warehouse-Regular-v0",
    ground_env_kwargs={"control_type": "ee", "render_mode": "human"},
)
obs, _ = env.reset()
returns = 0
rewards = []

for i in range(5000):
    ee_pos = obs[:3]

    gripper_action = np.array([-1.0])

    if env.u == 0:
        delta = np.array([0.0, 0.0, 0.1])
    elif env.u == 5:
        delta = cw.ABOVE_RED - ee_pos
    elif env.u == 4:
        delta = cw.GRASP_RED - ee_pos
    elif env.u == 3:
        delta = cw.GRASP_RED - ee_pos
    elif env.u == 2:
        delta = cw.RELEASE_RED - ee_pos
    elif env.u == 1:
        delta = cw.RELEASE_RED - ee_pos
    else:
        raise ValueError(f"Invalid state {env.u}")

    if env.u in (0, 5, 4, 1):
        # Open gripper
        gripper_action = np.array([1.0])
    else:
        # Close gripper
        gripper_action = np.array([-1.0])

    delta *= 2
    delta = np.clip(delta, -0.25, 0.25)

    action = np.concatenate([delta, gripper_action])

    obs, reward, terminated, truncated, _ = env.step(action)
    returns += reward
    rewards.append(reward)

    env.render()
    if terminated or truncated:
        print(f"Terminated at step {i}")
        break

rewards = np.array(rewards)
plt.plot(rewards)
plt.xlabel("Step")
plt.ylabel("Reward")
# plt.ylim(-3, 3)
plt.title("Reward for a sample trajectory within an episode")
plt.show()
