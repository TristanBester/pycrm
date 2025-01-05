import matplotlib.pyplot as plt
import numpy as np

from crm.agents.tabular.cql import CounterfactualQLearningAgent
from crm.agents.tabular.ql import QLearningAgent
from experiments.letter.lib.crossproduct import LetterWorldCrossProduct
from experiments.letter.lib.ground import LetterWorld
from experiments.letter.lib.label import LetterWorldLabellingFunction
from experiments.letter.lib.machine import LetterWorldCountingRewardMachine

if __name__ == "__main__":
    ground_env = LetterWorld()
    lf = LetterWorldLabellingFunction()
    crm = LetterWorldCountingRewardMachine()
    cross_product = LetterWorldCrossProduct(
        ground_env=ground_env,
        crm=crm,
        lf=lf,
        max_steps=300,
    )

    results_ql = []
    resu

    for _ in range(5):
        agent_ql = QLearningAgent(cross_product)

        all_returns_ql = agent_ql.learn(total_episodes=5000)

        smoothed_returns_ql = np.convolve(
            all_returns_ql, np.ones((300,)) / 300, mode="valid"
        )
        plt.plot(smoothed_returns_ql, label="QL")
    plt.legend()
    plt.show()

    # agent_cql = CounterfactualQLearningAgent(cross_product)
    # agent_ql = QLearningAgent(cross_product)

    # all_returns_cql = agent_cql.learn(total_episodes=5000)
    # all_returns_ql = agent_ql.learn(total_episodes=5000)

    # np.save("all_returns_cql.npy", all_returns_cql)
    # np.save("all_returns_ql.npy", all_returns_ql)

    # # all_returns_cql = np.load("all_returns_cql.npy")
    # # all_returns_ql = np.load("all_returns_ql.npy")

    # # Smooth using a rolling average
    # smoothed_returns_cql = np.convolve(
    #     all_returns_cql, np.ones((300,)) / 300, mode="valid"
    # )
    # smoothed_returns_ql = np.convolve(
    #     all_returns_ql, np.ones((300,)) / 300, mode="valid"
    # )
    # plt.plot(smoothed_returns_cql, label="CQL")
    # plt.plot(smoothed_returns_ql, label="QL")
    # plt.legend()
    # plt.show()
