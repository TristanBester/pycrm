import gymnasium as gym
from gymnasium.envs.registration import register

from experiments.letter.lib.crossproduct import LetterWorldCrossProduct
from experiments.letter.lib.ground import LetterWorld
from experiments.letter.lib.label import LetterWorldLabellingFunction
from experiments.letter.lib.machine import LetterWorldCountingRewardMachine

register(
    id="LetterWorld-v0",
    entry_point="experiments.letter.lib:make_letter_world_environment",
)


def make_letter_world_environment(**kwargs) -> gym.Env:
    """Create the letter world environment."""
    ground_env_kwargs = kwargs.get("ground_env_kwargs", {})
    crm_kwargs = kwargs.get("crm_kwargs", {})
    lf_kwargs = kwargs.get("lf_kwargs", {})
    crossproduct_kwargs = kwargs.get("crossproduct_kwargs", {})

    ground_env = LetterWorld(**ground_env_kwargs)
    lf = LetterWorldLabellingFunction(**lf_kwargs)
    crm = LetterWorldCountingRewardMachine(**crm_kwargs)
    cp = LetterWorldCrossProduct(
        ground_env=ground_env,
        crm=crm,
        lf=lf,
        **crossproduct_kwargs,
    )
    return cp
