import os

import gymnasium as gym
from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

import experiments.warehouse
from crm.agents.sb3.sac import CounterfactualSAC

# Allow the use of `pickle.load()` when downloading model from the hub
# Please make sure that the organization from which you download can be trusted
os.environ["TRUST_REMOTE_CODE"] = "True"

# Retrieve the model from the hub
## repo_id = id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name})
## filename = name of the model zip file from the repository
checkpoint = load_from_hub(
    repo_id="TristanBester/Warehouse-Order-Packer",
    filename="warehouse-policy.zip",
)
agent = CounterfactualSAC.load(checkpoint)

env = gym.make(
    "Warehouse-ContextSensitive-v0",
    ground_env_kwargs={"scene": "fancy", "control_type": "ee", "render_mode": "human"},
    crm_kwargs={},
    crossproduct_kwargs={"max_steps": 30000},
)


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
