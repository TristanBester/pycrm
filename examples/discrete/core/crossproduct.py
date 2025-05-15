import gymnasium as gym
import numpy as np

from crm.automaton import CountingRewardMachine
from crm.crossproduct import CrossProduct
from crm.label import LabellingFunction


class PuckWorldCrossProduct(CrossProduct[np.ndarray, np.ndarray, np.ndarray, None]):
    """Cross product of the Puck World environment."""

    def __init__(
        self,
        ground_env: gym.Env,
        crm: CountingRewardMachine,
        lf: LabellingFunction[np.ndarray, np.ndarray],
        max_steps: int,
    ) -> None:
        """Initialize the cross product Markov decision process environment."""
        super().__init__(ground_env, crm, lf, max_steps)
        self.observation_space = gym.spaces.Box(
            low=0, high=100, shape=(16,), dtype=np.int32
        )
        self.action_space = self.ground_env.action_space

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Step the environment."""
        obs, reward, terminated, truncated, info = super().step(action)

        if terminated or truncated:
            if self.steps >= self.max_steps:
                info["success"] = False
            else:
                info["success"] = True

        return obs, reward, terminated, truncated, info

    def _get_obs(
        self, ground_obs: np.ndarray, u: int, c: tuple[int, ...]
    ) -> np.ndarray:
        """Get the cross product observation.

        Args:
            ground_obs: The ground observation.
            u: The number of symbols seen.
            c: The counter configuration.

        Returns:
            Cross product observation - [ground obs, machine state, counter state].
        """
        crm_cfg = np.array([u, *c])
        return np.concatenate((ground_obs, crm_cfg))

    def to_ground_obs(self, obs: np.ndarray) -> np.ndarray:
        """Convert the cross product observation to a ground observation.

        Args:
            obs: The cross product observation.

        Returns:
            Ground observation.
        """
        return obs[:12]


class PuckWorldLoggingWrapper(gym.Wrapper):
    """Logging wrapper for the Puck World environment.

    Wrapper used to log subtask success rates during training.
    """

    def __init__(self, env: PuckWorldCrossProduct) -> None:
        """Initialize the logging wrapper."""
        super().__init__(env)
        self.t_1_1 = False
        self.t_1_2 = False
        self.t_1_3 = False

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        """Reset the environment."""
        assert isinstance(self.env, PuckWorldCrossProduct)
        obs, info = self.env.reset(**kwargs)
        self.u = self.env.u
        self.c = self.env.c

        self.t_1_1 = False
        self.t_1_2 = False
        self.t_1_3 = False

        subtask_info = self._get_subtask_info()
        info["subtask_info"] = subtask_info
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Step the environment."""
        assert isinstance(self.env, PuckWorldCrossProduct)
        last_u = self.env.u
        last_c = self.env.c

        obs, reward, terminated, truncated, info = self.env.step(action)
        curr_u = self.env.u
        curr_c = self.env.c

        self.u = self.env.u
        self.c = self.env.c

        if last_u != curr_u or last_c != curr_c:
            print(f"U: {last_u} -> {curr_u}\tC: {last_c} -> {curr_c}")

        self._update_subtask_info()
        subtask_info = self._get_subtask_info()

        info["subtask_info"] = subtask_info
        return obs, float(reward), terminated, truncated, info

    def _get_subtask_info(self) -> dict:
        """Get the subtask information.

        Returns:
            Subtask information.
        """
        return {
            "subtask/t_1_1_complete": int(self.t_1_1),
            "subtask/t_1_2_complete": int(self.t_1_2),
            "subtask/t_1_3_complete": int(self.t_1_3),
        }

    def _update_subtask_info(self) -> None:
        """Update the subtask information."""
        if self.u == 3 and self.c[0] == 0:
            self.t_1_1 = True
        if self.u == 0 and self.c[0] == 0:
            self.t_1_2 = True
        if self.u == 0 and self.c[0] == 1:
            self.t_1_3 = True
