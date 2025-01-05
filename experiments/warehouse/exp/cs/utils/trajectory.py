import os
import time
from warnings import filterwarnings

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import experiments.warehouse.lib.groundenv.constants.waypoints as cw
from crm.agents.sb3.sac import CounterfactualSAC

filterwarnings("ignore")


CHECKPOINT_PATH = "/Users/tristan/Projects/counting-reward-machines/checkpoints_old/model_18_5040000.zip"


env = gym.make(
    "Warehouse-ContextSensitive-v0",
    ground_env_kwargs={"control_type": "ee", "render_mode": "human", "scene": "fancy"},
    crm_kwargs={"c_0": (1, 1, 1)},
    crossproduct_kwargs={"max_steps": 10_000_000},
)

agent = CounterfactualSAC.load(CHECKPOINT_PATH)
obs, _ = env.reset()

positions = []

action_counter = 0
while True:
    action, _ = agent.predict(obs, deterministic=True)

    action *= 0
    obs, reward, terminated, truncated, _ = env.step(action)

    if terminated:
        print("Success at step", action_counter)
        break
    if truncated:
        print("Failed at step", action_counter)
        break
    action_counter += 1
    env.render()

positions = np.array(positions)


np.save("trajectory.npy", positions)
