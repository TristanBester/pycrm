import matplotlib.pyplot as plt
import numpy as np

from examples.rm.tabular.core import (
    OfficeWorld,
    OfficeWorldCrossProduct,
    OfficeWorldLabellingFunction,
    OfficeWorldRewardMachine,
)
from pycrm.agents.tabular import CounterfactualQLearningAgent, QLearningAgent

EPISODES = 500
LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.99
EPSILON = 0.1


def main():
    """Run the tabular experiment - compare QL and CQL agents."""
    # Initialize environment components
    ground_env = OfficeWorld()
    lf = OfficeWorldLabellingFunction()
    rm = OfficeWorldRewardMachine()
    cross_product = OfficeWorldCrossProduct(
        ground_env=ground_env,
        machine=rm,
        lf=lf,
        max_steps=500,
    )

    # Create a  Q-Learning agent
    ql_agent = QLearningAgent(
        env=cross_product,
        epsilon=EPSILON,
        learning_rate=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
    )
    # Create a Counterfactual Q-Learning agent
    cql_agent = CounterfactualQLearningAgent(
        env=cross_product,
        epsilon=EPSILON,
        learning_rate=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
    )

    # Train both agents & collect returns
    all_returns_ql = ql_agent.learn(total_episodes=EPISODES)
    all_returns_cql = cql_agent.learn(total_episodes=EPISODES)

    # Apply 100-episode moving average for smoother visualization
    smoothed_returns_ql = np.convolve(
        all_returns_ql, np.ones((100,)) / 100, mode="valid"
    )
    smoothed_returns_cql = np.convolve(
        all_returns_cql, np.ones((100,)) / 100, mode="valid"
    )

    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_returns_ql, label="Q-Learning")
    plt.plot(smoothed_returns_cql, label="Counterfactual Q-Learning")
    plt.title("Q-Learning Performance in Letter World Environment")
    plt.xlabel("Episode")
    plt.ylabel("Average Return (100-episode moving average)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    # Save visualization
    plt.savefig("office_world_experiment.png")
    plt.show()


if __name__ == "__main__":
    main()
