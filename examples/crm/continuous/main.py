import numpy as np
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.ddpg import DDPG
from stable_baselines3.sac import SAC
from stable_baselines3.td3 import TD3

from crm.agents.sb3.ddpg import CounterfactualDDPG

# from crm.agents.sb3.dqn import CounterfactualDQN
from crm.agents.sb3.sac import CounterfactualSAC
from crm.agents.sb3.td3 import CounterfactualTD3
from examples.crm.continuous.core import (
    PuckWorld,
    PuckWorldCountingRewardMachine,
    PuckWorldCrossProduct,
    PuckWorldLabellingFunction,
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
    ground_env = PuckWorld()
    lf = PuckWorldLabellingFunction()
    rm = PuckWorldCountingRewardMachine()
    cross_product = PuckWorldCrossProduct(
        ground_env=ground_env,
        machine=rm,
        lf=lf,
        max_steps=1000,
    )

    # agent = SAC(
    #     policy="MlpPolicy",
    #     env=cross_product,
    #     tensorboard_log="logs/",
    #     verbose=1,
    #     ent_coef=0.015,
    #     buffer_size=7_500_000,
    #     batch_size=2_500,
    #     learning_rate=0.001,
    #     tau=0.0009,
    # )
    # agent = CounterfactualSAC(
    #     policy="MlpPolicy",
    #     env=cross_product,
    #     tensorboard_log="logs/",
    #     verbose=1,
    #     ent_coef=0.015,
    #     buffer_size=7_500_000,
    #     batch_size=2_500,
    #     learning_rate=0.001,
    #     tau=0.0009,
    # )
    # agent = CounterfactualDDPG(
    #     policy="MlpPolicy",
    #     env=cross_product,
    #     tensorboard_log="logs/",
    #     verbose=1,
    #     buffer_size=1_000_000,
    #     batch_size=2_500,
    # )
    agent = DDPG(
        policy="MlpPolicy",
        env=cross_product,
        tensorboard_log="logs/",
        verbose=1,
        buffer_size=1_000_000,
        batch_size=2_500,
    )

    # agent = CounterfactualTD3(
    #     policy="MlpPolicy",
    #     env=cross_product,
    #     tensorboard_log="logs/",
    #     verbose=1,
    #     buffer_size=1_000_000,
    #     batch_size=2_500,
    # )
    # agent = TD3(
    #     policy="MlpPolicy",
    #     env=cross_product,
    #     tensorboard_log="logs/",
    #     verbose=1,
    #     buffer_size=1_000_000,
    #     batch_size=2_500,
    # )

    agent.learn(
        total_timesteps=200_000,
        log_interval=1,
    )


if __name__ == "__main__":
    main()
