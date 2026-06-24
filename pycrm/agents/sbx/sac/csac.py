from typing import Any

from stable_baselines3.common.type_aliases import MaybeCallback

from pycrm.agents.sbx.common import CounterfactualOffPolicyMixin

try:
    from sbx import SAC
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Counterfactual SBX agents require the optional SBX dependency. " +
        "Install it with `pip install pyrewardmachines[sbx]`."
    ) from exc


class CounterfactualSAC(CounterfactualOffPolicyMixin, SAC):
    """Counterfactual SAC using SBX's JAX implementation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the agent and counterfactual replay support."""
        super().__init__(*args, **kwargs)
        self._init_counterfactual_support()

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "SBX-C-SAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> "CounterfactualSAC":
        """Train the agent with a counterfactual TensorBoard name."""
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
