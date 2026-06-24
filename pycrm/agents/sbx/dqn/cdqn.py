from typing import Any

from stable_baselines3.common.type_aliases import MaybeCallback

from pycrm.agents.sbx.common import CounterfactualOffPolicyMixin
from pycrm.agents.sbx.dqn.dqn import DQN


class CounterfactualDQN(CounterfactualOffPolicyMixin, DQN):
    """Counterfactual DQN using SBX's JAX implementation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the agent and counterfactual replay support."""
        super().__init__(*args, **kwargs)
        self._init_counterfactual_support()

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "SBX-C-DQN",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> "CounterfactualDQN":
        """Train the agent with a counterfactual TensorBoard name."""
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
