import numpy as np
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.utils import safe_mean


class LoggingMixin(OffPolicyAlgorithm):
    """Add subtask logging."""

    def _dump_logs(self) -> None:
        """Add custom logging."""
        # Log all of the metrics tracked by sb3 by default
        super()._dump_logs()

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
