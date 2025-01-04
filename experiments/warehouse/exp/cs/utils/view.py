import time
from warnings import filterwarnings

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.env_util import make_vec_env

import experiments.warehouse.lib.groundenv.constants.waypoints as cw
from experiments.warehouse.lib.agents import LoggingCounterfactualSAC

filterwarnings("ignore")


CHECKPOINT_PATH = "/Users/tristan/Projects/counting-reward-machines/checkpoints_old/model_18_5040000.zip"

env = gym.make(
    "Warehouse-ContextSensitive-v0",
    ground_env_kwargs={"scene": "fancy", "control_type": "ee", "render_mode": "human"},
    crm_kwargs={},
    crossproduct_kwargs={"max_steps": 30000},
)

agent = LoggingCounterfactualSAC.load(CHECKPOINT_PATH, device="cpu")

obs, _ = env.reset()
last_u = -1
last_c = (-1, -1, -1)

u = env.unwrapped.u
c = env.unwrapped.c


for i in range(15000):
    action, _ = agent.predict(obs, deterministic=True)

    obs, reward, done, truncated, _ = env.step(action)

    last_u = u
    last_c = c

    u = env.unwrapped.u
    c = env.unwrapped.c

    if last_u != u or last_c != c:
        print(f"TRANSITION: {last_u} -> {u}\t{last_c} -> {c}")

        if u == 0 and c[0] != 0:
            env.unwrapped.ground_env.task.scene_manager.animate_red_block()
        elif u == 0 and c[0] == 0 and c[1] != 0:
            if c[1] == 3:
                env.unwrapped.ground_env.task.scene_manager.translate_tray("green")
            else:
                env.unwrapped.ground_env.task.scene_manager.animate_green_block()
        elif u == 0 and c[0] == 0 and c[1] == 0 and c[2] != 0:
            if c[2] == 3:
                env.unwrapped.ground_env.task.scene_manager.translate_tray("blue")
            else:
                env.unwrapped.ground_env.task.scene_manager.animate_blue_block()

    env.render()
    env.render()
    # time.sleep(0.0)
