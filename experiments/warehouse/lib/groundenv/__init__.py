import gymnasium as gym
from gymnasium.envs.registration import register

from experiments.warehouse.lib.groundenv.environment import PackCustomerOrderEnvironment
from experiments.warehouse.lib.groundenv.wrapper import EEStateWrapper


def make_warehouse_environment(**kwargs) -> gym.Env:
    """Factory function to create a warehouse environment."""
    env = PackCustomerOrderEnvironment(**kwargs)
    return EEStateWrapper(env)


register(
    id="WarehouseGround-v0",
    entry_point="experiments.warehouse.lib.groundenv:make_warehouse_environment",
)
