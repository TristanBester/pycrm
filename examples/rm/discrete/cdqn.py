import argparse

import wandb

from examples.rm.discrete.core import (
    PuckWorld,
    PuckWorldCrossProduct,
    PuckWorldLabellingFunction,
    PuckWorldRewardMachine,
)
from pycrm.agents.sb3.dqn import CounterfactualDQN


def main():
    """Run the tabular experiment - compare QL and CQL agents."""
    parser = argparse.ArgumentParser(
        description="Run DQN experiment with Reward Machines"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for the experiment",
        default=0,
    )
    args = parser.parse_args()

    wandb.init(
        project="PyCRM-Examples-RM-Discrete",
        name="C-DQN",
        sync_tensorboard=True,
    )

    # Initialize environment components
    ground_env = PuckWorld()
    lf = PuckWorldLabellingFunction()
    rm = PuckWorldRewardMachine()
    cross_product = PuckWorldCrossProduct(
        ground_env=ground_env,
        machine=rm,
        lf=lf,
        max_steps=1000,
    )

    # Initialize agent
    agent = CounterfactualDQN(
        policy="MlpPolicy",
        env=cross_product,
        tensorboard_log="logs/",
        verbose=1,
        exploration_fraction=0.2,
        exploration_final_eps=0.1,
        buffer_size=1_000_000,
        batch_size=2_500,
        device="cuda",
        seed=args.seed,
    )

    # Train agent
    agent.learn(
        total_timesteps=250_000,
        log_interval=1,
    )


if __name__ == "__main__":
    main()
