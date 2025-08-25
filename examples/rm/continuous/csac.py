import argparse

import wandb

from examples.rm.continuous.core import (
    PuckWorld,
    PuckWorldCrossProduct,
    PuckWorldLabellingFunction,
    PuckWorldRewardMachine,
)
from pycrm.agents.sb3.sac import CounterfactualSAC


def main():
    """Run the tabular experiment - compare QL and CQL agents."""
    parser = argparse.ArgumentParser(
        description="Run C-SAC experiment with Reward Machines"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for the experiment",
        default=0,
    )
    args = parser.parse_args()

    wandb.init(
        project="PyCRM-Examples-RM-Continuous",
        name="C-SAC",
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
        max_steps=500,
    )

    # Initialize agent
    agent = CounterfactualSAC(
        policy="MlpPolicy",
        env=cross_product,
        tensorboard_log="logs/",
        verbose=1,
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
