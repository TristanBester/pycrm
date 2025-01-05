import os
from warnings import filterwarnings

import gymnasium as gym
import hydra
import matplotlib.pyplot as plt
import numpy as np
import wandb
from omegaconf import DictConfig
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

from crm.agents.tabular.cql import CounterfactualQLearningAgent
from crm.agents.tabular.ql import QLearningAgent
from experiments.letter.lib.crossproduct import LetterWorldCrossProduct
from experiments.letter.lib.ground import LetterWorld
from experiments.letter.lib.label import LetterWorldLabellingFunction
from experiments.letter.lib.machine import LetterWorldCountingRewardMachine

filterwarnings("ignore")


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """Train a Q-learning agent."""
    results = []

    for seed in range(config.exp.n_runs):
        env = gym.make("LetterWorld-v0", seed=seed)
        agent = QLearningAgent(env)
        returns = agent.learn(total_episodes=config.exp.n_episodes)
        results.append(returns)

    output_dir = os.path.join(config.exp.log_dir, "results_ql.npy")
    np.save(output_dir, results)


if __name__ == "__main__":
    main()
