import glob
import os
import time

import gymnasium as gym
from stable_baselines3 import SAC

from experiments.warehouse.lib.crossproduct.crossproduct import WarehouseCrossProduct
from experiments.warehouse.lib.label.function import WarehouseLabellingFunction
from experiments.warehouse.lib.machine.machine import WarehouseCountingRewardMachine


def get_latest_checkpoint():
    """Get the most recent checkpoint file from the checkpoints directory."""
    checkpoint_files = glob.glob("./checkpoints/model_*.zip")
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint files found in ./checkpoints/")
    return max(checkpoint_files, key=os.path.getctime)


def create_env() -> WarehouseCrossProduct:
    """Create the warehouse environment."""
    ground_env = gym.make(
        "Warehouse-v0",
        render_mode="human",
    )
    crm = WarehouseCountingRewardMachine()
    lf = WarehouseLabellingFunction()
    env = WarehouseCrossProduct(ground_env, crm, lf)
    return env


def main() -> None:
    """Main function."""
    env = create_env()
    # Load the latest checkpoint
    checkpoint_path = get_latest_checkpoint()
    print(f"Loading model from: {checkpoint_path}")
    model = SAC.load(checkpoint_path)

    # Run 10 episodes
    for episode in range(10):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        print(f"\nStarting Episode {episode + 1}")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            # Optional: render the environment if available
            env.render()
            time.sleep(0.1)

        print(f"Episode {episode + 1} finished:")
        print(f"Total Reward: {total_reward}")
        print(f"Steps: {steps}")

    env.close()


if __name__ == "__main__":
    main()
