import gymnasium as gym
import numpy as np

from experiments.warehouse.lib.crossproduct import WarehouseCrossProduct


class RegularLoggingWrapper(gym.Wrapper):
    """Wrapper to add subtask information into 'info'."""

    def __init__(self, env: WarehouseCrossProduct) -> None:
        """Initialise the wrapper."""
        super().__init__(env)

        self._above_complete_3 = False
        self._above_complete_2 = False
        self._above_complete_1 = False

        self._grasp_complete_3 = False
        self._grasp_complete_2 = False
        self._grasp_complete_1 = False

        self._grip_complete_3 = False
        self._grip_complete_2 = False
        self._grip_complete_1 = False

        self._release_complete_3 = False
        self._release_complete_2 = False
        self._release_complete_1 = False

        self._drop_complete_3 = False
        self._drop_complete_2 = False
        self._drop_complete_1 = False

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

        match (self.env.u, self.env.c):
            case (4, (3,)):
                self._above_complete_3 = True
            case (4, (2,)):
                self._above_complete_2 = True
            case (4, (1,)):
                self._above_complete_1 = True
            case (3, (3,)):
                self._grasp_complete_3 = True
            case (3, (2,)):
                self._grasp_complete_2 = True
            case (3, (1,)):
                self._grasp_complete_1 = True
            case (2, (3,)):
                self._grip_complete_3 = True
            case (2, (2,)):
                self._grip_complete_2 = True
            case (2, (1,)):
                self._grip_complete_1 = True
            case (1, (3,)):
                self._release_complete_3 = True
            case (1, (2,)):
                self._release_complete_2 = True
            case (1, (1,)):
                self._release_complete_1 = True
            case (0, (2,)):
                self._drop_complete_3 = True
            case (0, (1,)):
                self._drop_complete_2 = True
            case (0, (0,)):
                self._drop_complete_1 = True

    def _get_subtask_info(self) -> dict:
        """Get the subtask information."""
        return {
            "above_complete_3": int(self._above_complete_3),
            "above_complete_2": int(self._above_complete_2),
            "above_complete_1": int(self._above_complete_1),
            "grasp_complete_3": int(self._grasp_complete_3),
            "grasp_complete_2": int(self._grasp_complete_2),
            "grasp_complete_1": int(self._grasp_complete_1),
            "grip_complete_3": int(self._grip_complete_3),
            "grip_complete_2": int(self._grip_complete_2),
            "grip_complete_1": int(self._grip_complete_1),
            "release_complete_3": int(self._release_complete_3),
            "release_complete_2": int(self._release_complete_2),
            "release_complete_1": int(self._release_complete_1),
            "drop_complete_3": int(self._drop_complete_3),
            "drop_complete_2": int(self._drop_complete_2),
            "drop_complete_1": int(self._drop_complete_1),
        }


class ContextFreeLoggingWrapper(gym.Wrapper):
    """Wrapper to add subtask information into 'info'."""

    def __init__(self, env: WarehouseCrossProduct) -> None:
        """Initialise the wrapper."""
        super().__init__(env)

        self._above_complete_3 = False
        self._above_complete_2 = False
        self._above_complete_1 = False

        self._grasp_complete_3 = False
        self._grasp_complete_2 = False
        self._grasp_complete_1 = False

        self._grip_complete_3 = False
        self._grip_complete_2 = False
        self._grip_complete_1 = False

        self._release_complete_3 = False
        self._release_complete_2 = False
        self._release_complete_1 = False

        self._drop_complete_3 = False
        self._drop_complete_2 = False
        self._drop_complete_1 = False

        self._red_complete_3 = False
        self._red_complete_2 = False
        self._red_complete_1 = False
        self._green_complete_3 = False
        self._green_complete_2 = False
        self._green_complete_1 = False

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

        match (self.env.u, self.env.c):
            case (9, (3, 3)):
                self._above_complete_3 = True
            case (9, (2, 3)):
                self._above_complete_2 = True
            case (9, (1, 3)):
                self._above_complete_1 = True
            case (8, (3, 3)):
                self._grasp_complete_3 = True
            case (8, (2, 3)):
                self._grasp_complete_2 = True
            case (8, (1, 3)):
                self._grasp_complete_1 = True
            case (7, (3, 3)):
                self._grip_complete_3 = True
            case (7, (2, 3)):
                self._grip_complete_2 = True
            case (7, (1, 3)):
                self._grip_complete_1 = True
            case (6, (3, 3)):
                self._release_complete_3 = True
            case (6, (2, 3)):
                self._release_complete_2 = True
            case (6, (1, 3)):
                self._release_complete_1 = True
            case (0, (2, 3)):
                self._drop_complete_3 = True
            case (0, (1, 3)):
                self._drop_complete_2 = True
            case (0, (0, 3)):
                self._drop_complete_1 = True

        match (self.env.u, self.env.c):
            case (0, (2, 3)):
                self._red_complete_3 = True
            case (0, (1, 3)):
                self._red_complete_2 = True
            case (0, (0, 3)):
                self._red_complete_1 = True
            case (0, (0, 2)):
                self._green_complete_3 = True
            case (0, (0, 1)):
                self._green_complete_2 = True
            case (0, (0, 0)):
                self._green_complete_1 = True

    def _get_subtask_info(self) -> dict:
        """Get the subtask information."""
        return {
            "above_complete_3": int(self._above_complete_3),
            "above_complete_2": int(self._above_complete_2),
            "above_complete_1": int(self._above_complete_1),
            "grasp_complete_3": int(self._grasp_complete_3),
            "grasp_complete_2": int(self._grasp_complete_2),
            "grasp_complete_1": int(self._grasp_complete_1),
            "grip_complete_3": int(self._grip_complete_3),
            "grip_complete_2": int(self._grip_complete_2),
            "grip_complete_1": int(self._grip_complete_1),
            "release_complete_3": int(self._release_complete_3),
            "release_complete_2": int(self._release_complete_2),
            "release_complete_1": int(self._release_complete_1),
            "drop_complete_3": int(self._drop_complete_3),
            "drop_complete_2": int(self._drop_complete_2),
            "drop_complete_1": int(self._drop_complete_1),
            "red_complete_3": int(self._red_complete_3),
            "red_complete_2": int(self._red_complete_2),
            "red_complete_1": int(self._red_complete_1),
            "green_complete_3": int(self._green_complete_3),
            "green_complete_2": int(self._green_complete_2),
            "green_complete_1": int(self._green_complete_1),
        }
