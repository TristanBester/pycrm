from enum import Enum, auto

import numpy as np

from crm.label import LabellingFunction


class Symbol(Enum):
    """Symbols in the Letter World environment."""

    A = auto()
    B = auto()
    C = auto()


class LetterWorldLabellingFunction(LabellingFunction[np.ndarray, int]):
    """Labelling function for the Letter World environment."""

    A_B_POSITION = np.array([1, 1])
    C_POSITION = np.array([1, 5])

    @LabellingFunction.event
    def test_symbol_a(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> Symbol | None:
        """Return if the agent observes symbol A."""
        if next_obs[0] == 0 and np.array_equal(next_obs[1:], self.A_B_POSITION):
            return Symbol.A
        return None

    @LabellingFunction.event
    def test_symbol_b(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> Symbol | None:
        """Return if the agent observes symbol B."""
        if next_obs[0] == 1 and np.array_equal(next_obs[1:], self.A_B_POSITION):
            return Symbol.B
        return None

    @LabellingFunction.event
    def test_symbol_c(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> Symbol | None:
        """Return if the agent observes symbol C."""
        if next_obs[0] == 1 and np.array_equal(next_obs[1:], self.C_POSITION):
            return Symbol.C
        return None
