import numpy as np
import sys
from torch import seed
import wandb
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.utils import safe_mean

from crm.agents.sb3.dqn import CounterfactualDQN
from examples.discrete.core import (
    PuckWorld,
    PuckWorldCountingRewardMachine,
    PuckWorldCrossProduct,
    PuckWorldLabellingFunction,
    PuckWorldLoggingWrapper,
)


class LoggingMixin(OffPolicyAlgorithm):
    """Add subtask logging."""

    def dump_logs(self) -> None:
        """Add custom logging."""
        # Log all of the metrics tracked by sb3 by default
        super().dump_logs()

        # Custom logging
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:  # type: ignore
            custom_keys = [k for k in self.ep_info_buffer[0].keys() if "complete" in k]  # type: ignore

            for key in custom_keys:
                self.logger.record(
                    f"subtask/{key}",
                    safe_mean(
                        [ep_info[key] for ep_info in self.ep_info_buffer]  # type: ignore
                    ),
                )

    def _update_info_buffer(
        self, infos: list[dict], dones: np.ndarray | None = None
    ) -> None:
        """Add custom episode info about subtask success rates."""
        super()._update_info_buffer(infos, dones)

        # Add custom information from the environment when it terminates
        for i, done in enumerate(dones):  # type: ignore
            if done:
                infos[i]["episode"].update(infos[i]["subtask_info"])


class LoggingCDQN(LoggingMixin, CounterfactualDQN):
    """DQN with subtask logging."""


def main():
    """Run the tabular experiment - compare QL and CQL agents."""
    wandb.init(
        project="crm-examples-discrete-v1",
        name="C-DQN",
        sync_tensorboard=True,
    )

    # Initialize environment components
    ground_env = PuckWorld()
    lf = PuckWorldLabellingFunction()
    crm = PuckWorldCountingRewardMachine()
    cross_product = PuckWorldCrossProduct(
        ground_env=ground_env,
        crm=crm,
        lf=lf,
        max_steps=1000,
    )
    env = PuckWorldLoggingWrapper(cross_product)

    agent = LoggingCDQN(
        policy="MlpPolicy",
        env=env,  # Must be a CrossProduct environment
        exploration_fraction=0.25,
        exploration_final_eps=0.1,
        tensorboard_log="logs/",
        verbose=1,
        learning_rate=0.0001,
        tau=1.0,
        target_update_interval=10000,
        buffer_size=1000000,
        learning_starts=1000,
        batch_size=32,
        seed=int(sys.argv[1]),
        device="cuda",
    )

    agent.learn(
        total_timesteps=5_000_000,
        log_interval=1,
    )


if __name__ == "__main__":
    main()
