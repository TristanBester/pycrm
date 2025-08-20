import argparse

import wandb
from stable_baselines3.ddpg import DDPG

from examples.crm.continuous.core import (
    PuckWorld,
    PuckWorldCountingRewardMachine,
    PuckWorldCrossProduct,
    PuckWorldLabellingFunction,
)


def main():
    """Run the tabular experiment - compare QL and CQL agents."""
    parser = argparse.ArgumentParser(
        description="Run DDPG experiment with Counting Reward Machines"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for the experiment",
        default=0,
    )
    args = parser.parse_args()

    wandb.init(
        project="PyCRM-Examples-CRM-Continuous",
        name="DDPG",
        sync_tensorboard=True,
    )

    # Initialize environment components
    ground_env = PuckWorld()
    lf = PuckWorldLabellingFunction()
    crm = PuckWorldCountingRewardMachine()
    cross_product = PuckWorldCrossProduct(
        ground_env=ground_env,
        machine=crm,
        lf=lf,
        max_steps=100,
    )

    # Initialize agent
    agent = DDPG(
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
        total_timesteps=50_000,
        log_interval=1,
    )


if __name__ == "__main__":
    main()
