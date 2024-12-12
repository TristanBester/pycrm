import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from experiments.warehouse.lib.crossproduct.crossproduct import WarehouseCrossProduct
from experiments.warehouse.lib.label.function import WarehouseLabellingFunction
from experiments.warehouse.lib.machine.machine import WarehouseCountingRewardMachine


def create_env() -> WarehouseCrossProduct:
    """Create the warehouse environment."""
    ground_env = gym.make(
        "Warehouse-v0",
    )
    crm = WarehouseCountingRewardMachine()
    lf = WarehouseLabellingFunction()
    env = WarehouseCrossProduct(ground_env, crm, lf)
    return env


def main() -> None:
    """Main function."""
    env = create_env()
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./checkpoints/",
        name_prefix="model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")
    model.learn(
        total_timesteps=1_000_000,
        callback=checkpoint_callback,
    )


if __name__ == "__main__":
    main()
