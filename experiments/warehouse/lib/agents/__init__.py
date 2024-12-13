from stable_baselines3 import SAC

from crm.agents.sb3.sac import CounterfactualSAC
from experiments.warehouse.lib.agents.logging import LoggingMixin


class LoggingSAC(LoggingMixin, SAC):
    """SAC with subtask logging."""


class LoggingCounterfactualSAC(LoggingMixin, CounterfactualSAC):
    """Counterfactual SAC with subtask logging."""
