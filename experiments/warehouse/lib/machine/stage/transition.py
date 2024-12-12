from dataclasses import dataclass

from experiments.warehouse.lib.machine.reward import RewardFunction


@dataclass
class Transition:
    """Stage of the block module."""

    formula: str
    current_state: int
    next_state: int
    counter_modifier: tuple[int, ...]
    reward_fn: RewardFunction
