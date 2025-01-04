import os
import time
from itertools import product
from warnings import filterwarnings

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.env_util import make_vec_env

import experiments.warehouse.lib.groundenv.constants.waypoints as cw
from experiments.warehouse.lib.agents import LoggingCounterfactualSAC

filterwarnings("ignore")


CHECKPOINT_DIR = "/Users/tristan/Projects/counting-reward-machines/checkpoints_old"
CHECKPOINTS = [
    "model_1_5310000.zip",
    "model_8_5280000.zip",
    "model_18_5040000.zip",
    "model_0_5010000.zip",
    "model_5_5100000.zip",
    "model_3_5250000.zip",
    "model_3_5130000.zip",
    "model_16_5010000.zip",
    "model_1_5280000.zip",
    "model_5_5190000.zip",
    "model_4_5160000.zip",
    "model_11_5040000.zip",
    "model_7_5220000.zip",
    "model_4_5280000.zip",
    "model_13_5010000.zip",
    "model_2_5220000.zip",
    "model_4_5070000.zip",
    "model_14_5040000.zip",
    "model_1_5160000.zip",
    "model_0_5190000.zip",
    "model_12_5070000.zip",
    "model_4_5310000.zip",
    "model_2_5040000.zip",
    "model_0_5100000.zip",
    "model_8_5160000.zip",
    "model_5_5010000.zip",
    "model_5_5250000.zip",
    "model_3_5100000.zip",
    "model_7_5310000.zip",
]


for checkpoint in CHECKPOINTS:
    print("#" * 100)
    print(f"EVALUATING: {checkpoint}")
    print("#" * 100)

    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint)
    agent = LoggingCounterfactualSAC.load(checkpoint_path, device="cpu")

    solution_counter = 0
    failed = False

    for c_0 in list(product(range(4), repeat=3))[::-1]:
        env = gym.make(
            "Warehouse-ContextSensitive-v0",
            ground_env_kwargs={"control_type": "ee", "render_mode": "rgb_array"},
            crm_kwargs={"c_0": c_0},
            crossproduct_kwargs={"max_steps": 15000},
        )

        obs, _ = env.reset()
        for _ in range(15001):
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)

            if terminated:
                solution_counter += 1
                break
            if truncated:
                failed = True
                break

        if failed:
            break

    print(checkpoint, solution_counter)
