import gymnasium as gym
import numpy as np

from experiments.warehouse.lib.crossproduct import WarehouseCrossProduct


class LoggingWrapper(gym.Wrapper):
    """Wrapper to add subtask information into 'info'."""

    def __init__(self, env: WarehouseCrossProduct) -> None:
        """Initialise the wrapper."""
        super().__init__(env)

        self._above_complete = False
        self._grasp_complete = False
        self._grip_complete = False
        self._release_complete = False
        self._drop_complete = False

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        """Reset the environment."""
        assert isinstance(self.env, WarehouseCrossProduct)

        obs, info = self.env.reset(**kwargs)
        self.u = self.env.u
        self.c = self.env.c

        subtask_info = self._get_subtask_info()
        info["subtask_info"] = subtask_info
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Step the environment."""
        assert isinstance(self.env, WarehouseCrossProduct)

        last_u = self.env.u
        last_c = self.env.c
        obs, reward, terminated, truncated, info = self.env.step(action)
        curr_u = self.env.u
        curr_c = self.env.c

        self.u = self.env.u
        self.c = self.env.c

        if last_u != curr_u or last_c != curr_c:
            print(f"{last_u} -> {curr_u}\t{last_c} -> {curr_c}")

        self._update_subtask_info()
        subtask_info = self._get_subtask_info()

        info["subtask_info"] = subtask_info
        return obs, float(reward), terminated, truncated, info

    def _update_subtask_info(self) -> None:
        """Update the subtask information."""
        assert isinstance(self.env, WarehouseCrossProduct)

        match self.env.u:
            case 4:
                self._above_complete = True
            case 3:
                self._grasp_complete = True
            case 2:
                self._grip_complete = True
            case 1:
                self._release_complete = True

        match self.env.c:
            case (0,):
                self._drop_complete = True

    def _get_subtask_info(self) -> dict:
        """Get the subtask information."""
        return {
            "above_complete": int(self._above_complete),
            "grasp_complete": int(self._grasp_complete),
            "grip_complete": int(self._grip_complete),
            "release_complete": int(self._release_complete),
            "drop_complete": int(self._drop_complete),
        }
