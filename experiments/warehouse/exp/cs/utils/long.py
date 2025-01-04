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

"""
12 -> 11        (1, 0, 0) -> (1, 0, 0)
11 -> 0 (1, 0, 0) -> (0, 0, 0)
0 -> 16 (0, 0, 0) -> (0, 0, 0)
Success at step 3,153,117
"""

CHECKPOINT_PATH = "/Users/tristan/Projects/counting-reward-machines/checkpoints_old/model_18_5040000.zip"


env = gym.make(
    "Warehouse-ContextSensitive-v0",
    ground_env_kwargs={"control_type": "ee", "render_mode": "rgb_array"},
    crm_kwargs={"c_0": (2000, 0, 0)},
    crossproduct_kwargs={"max_steps": 10_000_000},
)

agent = CounterfactualSAC.load(CHECKPOINT_PATH)
obs, _ = env.reset()

action_counter = 0
while True:
    action, _ = agent.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)

    if terminated:
        print("Success at step", action_counter)
        break
    if truncated:
        print("Failed at step", action_counter)
        break
    action_counter += 1
