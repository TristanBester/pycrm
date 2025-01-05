import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id="LetterWorld-v0",
    entry_point="experiments.warehouse.lib:make_context_sensitive_warehouse_environment",
)


def make_letter_world_environment(**kwargs) -> gym.Env:
    """Create the warehouse environment."""
    ground_env_kwargs = kwargs.get("ground_env_kwargs", {})
    crm_kwargs = kwargs.get("crm_kwargs", {})
    lf_kwargs = kwargs.get("lf_kwargs", {})
    crossproduct_kwargs = kwargs.get("crossproduct_kwargs", {})

    ground_env = LetterWorld(**ground_env_kwargs)
    lf = LetterWorldLabellingFunction(**lf_kwargs)
    crm = LetterWorldCountingRewardMachine(**crm_kwargs)
    cross_product = LetterWorldCrossProduct(
        ground_env=ground_env,
        crm=crm,
        lf=lf,
        max_steps=300,
    )

    ground_env = gym.make("WarehouseGround-v0", **ground_env_kwargs)
    crm = RegularCRM(**crm_kwargs)
    lf = WarehouseLabellingFunction(**lf_kwargs)
    cp = WarehouseCrossProduct(ground_env, crm, lf, **crossproduct_kwargs)
    env = RegularLoggingWrapper(env=cp)
    return env
