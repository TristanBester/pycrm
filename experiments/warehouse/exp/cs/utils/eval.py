import os
import time
from warnings import filterwarnings

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import experiments.warehouse.lib.groundenv.constants.waypoints as cw
from crm.agents.sb3.sac import CounterfactualSAC

filterwarnings("ignore")


CHECKPOINT_DIR = "/Users/tristan/Projects/counting-reward-machines/checkpoints"


env = gym.make(
    "Warehouse-ContextSensitive-v0",
    ground_env_kwargs={"control_type": "ee", "render_mode": "rgb_array"},
    crossproduct_kwargs={"max_steps": 30000},
)


successes = []


for filename in tqdm(os.listdir(CHECKPOINT_DIR)):
    print("SUCCESS:", successes)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, filename)

    agent = CounterfactualSAC.load(checkpoint_path)

    obs, _ = env.reset()

    for x in range(30000):
        action, _ = agent.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, _ = env.step(action)

        if terminated:
            print("Success", filename)
            successes.append(filename)
            break
        if truncated:
            print("Failed at step", x, " - ", filename)
            break
    print()

print("SUCCESS:", successes)
