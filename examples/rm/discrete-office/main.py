import numpy as np
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.dqn import DQN

from crm.agents.sb3.dqn import CounterfactualDQN
from examples.rm.discrete.core import (
    OfficeWorld,
    OfficeWorldCrossProduct,
    OfficeWorldLabellingFunction,
    OfficeWorldRewardMachine,
)

# class LoggingMixin(OffPolicyAlgorithm):
#     """Add subtask logging."""

#     def dump_logs(self) -> None:
#         """Add custom logging."""
#         # Log all of the metrics tracked by sb3 by default
#         super().dump_logs()

#         # Custom logging
#         if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:  # type: ignore
#             custom_keys = [k for k in self.ep_info_buffer[0].keys() if "complete" in k]  # type: ignore

#             for key in custom_keys:
#                 self.logger.record(
#                     f"subtask/{key}",
#                     safe_mean(
#                         [ep_info[key] for ep_info in self.ep_info_buffer]  # type: ignore
#                     ),
#                 )

#     def _update_info_buffer(
#         self, infos: list[dict], dones: np.ndarray | None = None
#     ) -> None:
#         """Add custom episode info about subtask success rates."""
#         super()._update_info_buffer(infos, dones)

#         # Add custom information from the environment when it terminates
#         for i, done in enumerate(dones):  # type: ignore
#             if done:
#                 infos[i]["episode"].update(infos[i]["subtask_info"])


# class LoggingDQN(LoggingMixin, DQN):
#     """DQN with subtask logging."""


def main():
    """Run the tabular experiment - compare QL and CQL agents."""
    # Initialize environment components
    ground_env = OfficeWorld()
    lf = OfficeWorldLabellingFunction()
    rm = OfficeWorldRewardMachine()
    cross_product = OfficeWorldCrossProduct(
        ground_env=ground_env,
        machine=rm,
        lf=lf,
        max_steps=1000,
    )

    agent = DQN(
        policy="MlpPolicy",
        env=cross_product,  # Must be a CrossProduct environment
        exploration_fraction=0.5,
        exploration_final_eps=0.1,
        tensorboard_log="logs/",
        verbose=1,
        learning_rate=0.001,
        tau=1.0,
        target_update_interval=10000,
        buffer_size=1000000,
        learning_starts=1000,
        batch_size=1024,
    )
    agent = DQN(
        policy="MlpPolicy",
        env=cross_product,  # Must be a CrossProduct environment
        exploration_fraction=0.5,
        exploration_final_eps=0.1,
        tensorboard_log="logs/",
        verbose=1,
        learning_rate=0.001,
        tau=1.0,
        target_update_interval=10000,
        buffer_size=1000000,
        learning_starts=1000,
        batch_size=1024,
    )

    agent.learn(
        total_timesteps=200_000,
        log_interval=10,
    )


if __name__ == "__main__":
    main()
