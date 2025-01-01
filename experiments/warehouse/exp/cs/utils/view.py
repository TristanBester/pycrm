import time
from warnings import filterwarnings

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.env_util import make_vec_env

import experiments.warehouse.lib.groundenv.constants.waypoints as cw
from crm.agents.sb3.sac import CounterfactualSAC

filterwarnings("ignore")


CHECKPOINT_PATH = (
    "/Users/tristan/Projects/counting-reward-machines/checkpoints/model_0_2610000.zip"
)

env = gym.make(
    "Warehouse-ContextSensitive-v0",
    ground_env_kwargs={"control_type": "ee", "render_mode": "human"},
    crm_kwargs={},
    crossproduct_kwargs={"max_steps": 30000},
)

# agent = CounterfactualSAC.load(CHECKPOINT_PATH)

# obs, _ = env.reset()

# for i in range(15000):
#     action, _ = agent.predict(obs, deterministic=False)

#     # action[:3] = action[:3] + np.random.normal(0, 0.5, (3,))
#     # action[:3] = np.clip(action[:3], -0.025, 0.025)
#     obs, reward, done, truncated, _ = env.step(action)

#     env.render()
#     # time.sleep(0.0)

model = CounterfactualSAC.load(CHECKPOINT_PATH, env=env)


# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
