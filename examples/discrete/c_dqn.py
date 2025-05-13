from crm.agents.sb3.dqn import CounterfactualDQN
from examples.discrete.core import (
    PuckWorld,
    PuckWorldCountingRewardMachine,
    PuckWorldCrossProduct,
    PuckWorldLabellingFunction,
)

EPISODES = 5000
LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.99
EPSILON = 0.1


def main():
    """Run the tabular experiment - compare QL and CQL agents."""
    # Initialize environment components
    ground_env = PuckWorld()
    lf = PuckWorldLabellingFunction()
    crm = PuckWorldCountingRewardMachine()
    cross_product = PuckWorldCrossProduct(
        ground_env=ground_env,
        crm=crm,
        lf=lf,
        max_steps=500,
    )

    agent = CounterfactualDQN(
        policy="MlpPolicy",
        env=cross_product,  # Must be a CrossProduct environment
        exploration_fraction=0.25,
        tensorboard_log="logs/",
        verbose=1,
    )

    agent.learn(total_timesteps=1_000_000, log_interval=50)


if __name__ == "__main__":
    main()
