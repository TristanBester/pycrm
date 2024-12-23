from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.utils import safe_mean

from crm.agents.sb3.sac import CounterfactualSAC


# FIXME: consider moving this to a more general subtask implementation which can be used for all CSAC agents
class ContextSensitiveSubtaskLoggingCSAC(CounterfactualSAC):
    """
    The whole purpose of this class is to provide subtask logging.
    """

    def _dump_logs(self) -> None:
        """Add custom logging."""
        # Log all of the metrics tracked by sb3 by default
        super()._dump_logs()

        # Custom logging
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:  # type: ignore
            self.logger.record(
                "subtask/green_one_complete",
                safe_mean(
                    [ep_info["green_one_complete"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "subtask/green_two_complete",
                safe_mean(
                    [ep_info["green_two_complete"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "subtask/green_three_complete",
                safe_mean(
                    [ep_info["green_three_complete"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "subtask/red_one_complete",
                safe_mean(
                    [ep_info["red_one_complete"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "subtask/red_two_complete",
                safe_mean(
                    [ep_info["red_two_complete"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "subtask/red_three_complete",
                safe_mean(
                    [ep_info["red_three_complete"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "subtask/blue_one_complete",
                safe_mean(
                    [ep_info["blue_one_complete"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "subtask/blue_two_complete",
                safe_mean(
                    [ep_info["blue_two_complete"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "subtask/blue_three_complete",
                safe_mean(
                    [ep_info["blue_three_complete"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )

            self.logger.record(
                "stage-green/green_above_3_0",
                safe_mean(
                    [ep_info["green_above_3_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-green/green_above_2_0",
                safe_mean(
                    [ep_info["green_above_2_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-green/green_above_1_0",
                safe_mean(
                    [ep_info["green_above_1_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-green/green_grasp_3_0",
                safe_mean(
                    [ep_info["green_grasp_3_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-green/green_grasp_2_0",
                safe_mean(
                    [ep_info["green_grasp_2_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-green/green_grasp_1_0",
                safe_mean(
                    [ep_info["green_grasp_1_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-green/green_release_3_0",
                safe_mean(
                    [ep_info["green_release_3_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-green/green_release_2_0",
                safe_mean(
                    [ep_info["green_release_2_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-green/green_release_1_0",
                safe_mean(
                    [ep_info["green_release_1_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-red/red_above_3_0",
                safe_mean(
                    [ep_info["red_above_3_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-red/red_above_2_0",
                safe_mean(
                    [ep_info["red_above_2_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-red/red_above_1_0",
                safe_mean(
                    [ep_info["red_above_1_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-red/red_grasp_3_0",
                safe_mean(
                    [ep_info["red_grasp_3_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-red/red_grasp_2_0",
                safe_mean(
                    [ep_info["red_grasp_2_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-red/red_grasp_1_0",
                safe_mean(
                    [ep_info["red_grasp_1_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-red/red_release_3_0",
                safe_mean(
                    [ep_info["red_release_3_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-red/red_release_2_0",
                safe_mean(
                    [ep_info["red_release_2_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-red/red_release_1_0",
                safe_mean(
                    [ep_info["red_release_1_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-blue/blue_above_3_0",
                safe_mean(
                    [ep_info["blue_above_3_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-blue/blue_above_2_0",
                safe_mean(
                    [ep_info["blue_above_2_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-blue/blue_above_1_0",
                safe_mean(
                    [ep_info["blue_above_1_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-blue/blue_grasp_3_0",
                safe_mean(
                    [ep_info["blue_grasp_3_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-blue/blue_grasp_2_0",
                safe_mean(
                    [ep_info["blue_grasp_2_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-blue/blue_grasp_1_0",
                safe_mean(
                    [ep_info["blue_grasp_1_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-blue/blue_release_3_0",
                safe_mean(
                    [ep_info["blue_release_3_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-blue/blue_release_2_0",
                safe_mean(
                    [ep_info["blue_release_2_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-blue/blue_release_1_0",
                safe_mean(
                    [ep_info["blue_release_1_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )

    def _update_info_buffer(
        self, infos: List[Dict[str, Any]], dones: Optional[np.ndarray] = None
    ) -> None:
        """Add custom episode info about the CRA state."""
        super()._update_info_buffer(infos, dones)

        # Add custom information from the environment when it terminates
        for i, done in enumerate(dones):  # type: ignore
            if done:
                infos[i]["episode"].update(infos[i]["subtask"])
                infos[i]["episode"].update(infos[i]["stage"])


# FIXME: consider moving this to a more general subtask implementation which can be used for all CSAC agents
class ContextSensitiveSubtaskLoggingSAC(SAC):
    """
    The whole purpose of this class is to provide subtask logging.
    """

    def _dump_logs(self) -> None:
        """Add custom logging."""
        # Log all of the metrics tracked by sb3 by default
        super()._dump_logs()

        # Custom logging
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:  # type: ignore
            self.logger.record(
                "subtask/green_one_complete",
                safe_mean(
                    [ep_info["green_one_complete"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "subtask/green_two_complete",
                safe_mean(
                    [ep_info["green_two_complete"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "subtask/green_three_complete",
                safe_mean(
                    [ep_info["green_three_complete"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "subtask/red_one_complete",
                safe_mean(
                    [ep_info["red_one_complete"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "subtask/red_two_complete",
                safe_mean(
                    [ep_info["red_two_complete"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "subtask/red_three_complete",
                safe_mean(
                    [ep_info["red_three_complete"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "subtask/blue_one_complete",
                safe_mean(
                    [ep_info["blue_one_complete"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "subtask/blue_two_complete",
                safe_mean(
                    [ep_info["blue_two_complete"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "subtask/blue_three_complete",
                safe_mean(
                    [ep_info["blue_three_complete"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )

            self.logger.record(
                "stage-green/green_above_3_0",
                safe_mean(
                    [ep_info["green_above_3_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-green/green_above_2_0",
                safe_mean(
                    [ep_info["green_above_2_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-green/green_above_1_0",
                safe_mean(
                    [ep_info["green_above_1_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-green/green_grasp_3_0",
                safe_mean(
                    [ep_info["green_grasp_3_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-green/green_grasp_2_0",
                safe_mean(
                    [ep_info["green_grasp_2_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-green/green_grasp_1_0",
                safe_mean(
                    [ep_info["green_grasp_1_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-green/green_release_3_0",
                safe_mean(
                    [ep_info["green_release_3_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-green/green_release_2_0",
                safe_mean(
                    [ep_info["green_release_2_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-green/green_release_1_0",
                safe_mean(
                    [ep_info["green_release_1_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-red/red_above_3_0",
                safe_mean(
                    [ep_info["red_above_3_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-red/red_above_2_0",
                safe_mean(
                    [ep_info["red_above_2_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-red/red_above_1_0",
                safe_mean(
                    [ep_info["red_above_1_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-red/red_grasp_3_0",
                safe_mean(
                    [ep_info["red_grasp_3_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-red/red_grasp_2_0",
                safe_mean(
                    [ep_info["red_grasp_2_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-red/red_grasp_1_0",
                safe_mean(
                    [ep_info["red_grasp_1_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-red/red_release_3_0",
                safe_mean(
                    [ep_info["red_release_3_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-red/red_release_2_0",
                safe_mean(
                    [ep_info["red_release_2_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-red/red_release_1_0",
                safe_mean(
                    [ep_info["red_release_1_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-blue/blue_above_3_0",
                safe_mean(
                    [ep_info["blue_above_3_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-blue/blue_above_2_0",
                safe_mean(
                    [ep_info["blue_above_2_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-blue/blue_above_1_0",
                safe_mean(
                    [ep_info["blue_above_1_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-blue/blue_grasp_3_0",
                safe_mean(
                    [ep_info["blue_grasp_3_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-blue/blue_grasp_2_0",
                safe_mean(
                    [ep_info["blue_grasp_2_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-blue/blue_grasp_1_0",
                safe_mean(
                    [ep_info["blue_grasp_1_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-blue/blue_release_3_0",
                safe_mean(
                    [ep_info["blue_release_3_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-blue/blue_release_2_0",
                safe_mean(
                    [ep_info["blue_release_2_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )
            self.logger.record(
                "stage-blue/blue_release_1_0",
                safe_mean(
                    [ep_info["blue_release_1_0"] for ep_info in self.ep_info_buffer]  # type: ignore
                ),  # type: ignore
            )

    def _update_info_buffer(
        self, infos: List[Dict[str, Any]], dones: Optional[np.ndarray] = None
    ) -> None:
        """Add custom episode info about the CRA state."""
        super()._update_info_buffer(infos, dones)

        # Add custom information from the environment when it terminates
        for i, done in enumerate(dones):  # type: ignore
            if done:
                infos[i]["episode"].update(infos[i]["subtask"])
                infos[i]["episode"].update(infos[i]["stage"])
