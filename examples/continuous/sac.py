import numpy as np
import sys
import wandb
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.sac import SAC

from examples.continuous.core import (
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


class LoggingSAC(LoggingMixin, SAC):
    """SAC with subtask logging."""


def main():
    """Run the tabular experiment - compare QL and CQL agents."""
    wandb.init(
        project="crm-examples-v1",
        name="SAC",
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
        max_steps=100,
    )
    env = PuckWorldLoggingWrapper(cross_product)

    agent = LoggingSAC(
        policy="MlpPolicy",
        env=env,  # Must be a CrossProduct environment
        tensorboard_log="logs/",
        verbose=1,
        ent_coef=0.015,
        buffer_size=7_500_000,
        batch_size=2_500,
        learning_rate=0.001,
        tau=0.0009,
        device="cuda",
        seed=int(sys.argv[1]),
    )

    agent.learn(
        total_timesteps=5_000_000,
        log_interval=1,
    )


if __name__ == "__main__":
    main()
