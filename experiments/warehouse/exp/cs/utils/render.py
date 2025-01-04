import time
from warnings import filterwarnings

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3.common.env_util import make_vec_env

import experiments.warehouse.lib.groundenv.constants.waypoints as cw
from experiments.warehouse.lib.agents import LoggingCounterfactualSAC

filterwarnings("ignore")


CHECKPOINT_PATH = "/Users/tristan/Projects/counting-reward-machines/checkpoints_old/model_18_5040000.zip"


for c_0 in [(2, 2, 0), (2, 0, 2), (0, 2, 2)]:
    env = gym.make(
        "Warehouse-ContextSensitive-v0",
        ground_env_kwargs={
            "scene": "fancy",
            "control_type": "ee",
            "render_mode": "human",
        },
        crm_kwargs={"c_0": c_0},
        crossproduct_kwargs={"max_steps": 30000},
    )

    agent = LoggingCounterfactualSAC.load(CHECKPOINT_PATH, device="cpu")

    obs, _ = env.reset()
    last_u = -1
    last_c = (-1, -1, -1)

    u = env.unwrapped.u
    c = env.unwrapped.c

    r_max = c_0[0]
    g_max = c_0[1]
    b_max = c_0[2]

    moved_tray_green = False
    moved_tray_blue = False

    if u == 0:
        if c[0] == 0:
            if not moved_tray_green:
                env.unwrapped.ground_env.task.scene_manager.translate_tray("green")
                moved_tray_green = True
        if c[0] == 0 and c[1] == 0:
            if not moved_tray_blue:
                env.unwrapped.ground_env.task.scene_manager.translate_tray("blue")
                moved_tray_blue = True
        if c[0] == 0 and c[1] == 0 and c[2] == 0:
            if not moved_tray_red:
                env.unwrapped.ground_env.task.scene_manager.translate_tray("red")
                moved_tray_red = True

    values = []
    rewards = []
    for i in range(15000):
        action, _ = agent.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, _ = env.step(action)

        obs_t = torch.tensor(obs).reshape(1, -1)
        action_t = torch.tensor(action).reshape(1, -1)

        value = agent.critic(obs_t, action_t)

        values.append(value[0].item())
        rewards.append(reward)

        last_u = u
        last_c = c

        u = env.unwrapped.u
        c = env.unwrapped.c

        if last_u != u or last_c != c:
            print(f"TRANSITION: {last_u} -> {u}\t{last_c} -> {c}")

            if u == 0:
                if c[0] == 0:
                    if not moved_tray_green:
                        env.unwrapped.ground_env.task.scene_manager.translate_tray(
                            "green"
                        )
                        moved_tray_green = True
                if c[0] == 0 and c[1] == 0:
                    if not moved_tray_blue:
                        env.unwrapped.ground_env.task.scene_manager.translate_tray(
                            "blue"
                        )
                        moved_tray_blue = True

            if u == 0 and c[0] != 0 and c[0] != r_max:
                env.unwrapped.ground_env.task.scene_manager.animate_red_block()
            elif u == 0 and c[0] == 0 and c[1] != 0 and c[1] != g_max:
                env.unwrapped.ground_env.task.scene_manager.animate_green_block()
            elif u == 0 and c[0] == 0 and c[1] == 0 and c[2] != 0 and c[2] != b_max:
                env.unwrapped.ground_env.task.scene_manager.animate_blue_block()

        if terminated:
            break

        env.render()

    plt.plot(values)
    plt.show()

    plt.plot(rewards)
    plt.show()

    env.close()
    del env
    break

    time.sleep(1.0)
