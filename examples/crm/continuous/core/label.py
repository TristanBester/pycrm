from enum import Enum, auto

import numpy as np

from pycrm.label import LabellingFunction


class Symbol(Enum):
    """Symbols in the Puck World environment."""

    T_1 = auto()  # Target 1
    T_2 = auto()  # Target 2
    T_3 = auto()  # Target 3
    A = auto()  # Adversary
    DEFAULT = auto()  # Default


class PuckWorldLabellingFunction(LabellingFunction[np.ndarray, np.ndarray]):
    """Labelling function for the Puck World environment."""

    TARGET_THRESHOLD = 0.15
    ADVERSARY_THRESHOLD = 0.1

    @LabellingFunction.event
    def test_target_one(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> Symbol | None:
        """Test if the agent is tracking target one."""
        del obs, action

        agent_pos, _, target_one_pos, _, _, _ = self._unpack_obs(next_obs)
        dist = np.linalg.norm(agent_pos - target_one_pos)
        if dist < self.TARGET_THRESHOLD:
            return Symbol.T_1
        return None

    @LabellingFunction.event
    def test_target_two(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> Symbol | None:
        """Test if the agent is tracking target two."""
        del obs, action

        agent_pos, _, _, target_two_pos, _, _ = self._unpack_obs(next_obs)
        dist = np.linalg.norm(agent_pos - target_two_pos)
        if dist < self.TARGET_THRESHOLD:
            return Symbol.T_2
        return None

    @LabellingFunction.event
    def test_target_three(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> Symbol | None:
        """Test if the agent is tracking target three."""
        del obs, action

        agent_pos, _, _, _, target_three_pos, _ = self._unpack_obs(next_obs)
        dist = np.linalg.norm(agent_pos - target_three_pos)
        if dist < self.TARGET_THRESHOLD:
            return Symbol.T_3
        return None

    @LabellingFunction.event
    def test_adversary(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> Symbol | None:
        """Test if the agent is tracking the adversary."""
        del obs, action

        agent_pos, _, _, _, _, adversary_pos = self._unpack_obs(next_obs)
        dist = np.linalg.norm(agent_pos - adversary_pos)
        if dist < self.ADVERSARY_THRESHOLD:
            return Symbol.A
        return None

    @LabellingFunction.event
    def test_default(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> Symbol | None:
        """Return the default symbol (always true)."""
        del obs, action, next_obs
        return Symbol.DEFAULT

    def _unpack_obs(self, obs: np.ndarray) -> tuple[np.ndarray, ...]:
        agent_pos = obs[:2]
        agent_vel = obs[2:4]
        target_one_pos = obs[4:6]
        target_two_pos = obs[6:8]
        target_three_pos = obs[8:10]
        adversary_pos = obs[10:12]
        return (
            agent_pos,
            agent_vel,
            target_one_pos,
            target_two_pos,
            target_three_pos,
            adversary_pos,
        )
