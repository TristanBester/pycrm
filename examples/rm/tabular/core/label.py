from enum import Enum, auto

import numpy as np

from pycrm.label import LabellingFunction


class Symbol(Enum):
    """Symbols in the Letter World environment."""

    C = auto()  # Coffee machine
    M = auto()  # Mail Collected
    E = auto()  # Mail Empty
    P = auto()  # People
    D = auto()  # Decoration


class OfficeWorldLabellingFunction(LabellingFunction[np.ndarray, int]):
    """Labelling function for the Office World environment."""

    COFFEE_COORDS = np.array([2, 6])
    MAIL_COORDS = np.array([6, 10])
    PEOPLE_COORDS = np.array([6, 6])
    DECORATION_COORD_LIST = np.array([[6, 2], [6, 14], [10, 6], [10, 10]])

    @LabellingFunction.event
    def test_coffee_machine(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> Symbol | None:
        """Return if the agent is at the coffee machine."""
        del obs, action

        if np.array_equal(next_obs[:2], self.COFFEE_COORDS):
            return Symbol.C
        return None

    @LabellingFunction.event
    def test_mail(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> Symbol | None:
        """Return if the agent is at the mail."""
        del obs, action

        if np.array_equal(next_obs[:2], self.MAIL_COORDS):
            if next_obs[2] == 0:
                return Symbol.M
            elif next_obs[2] == 1:
                return Symbol.E
            else:
                raise ValueError("Invalid observation.")
        return None

    @LabellingFunction.event
    def test_people(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> Symbol | None:
        """Return if the agent is at the people."""
        del obs, action

        if np.array_equal(next_obs[:2], self.PEOPLE_COORDS):
            return Symbol.P
        return None

    @LabellingFunction.event
    def test_decoration(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> Symbol | None:
        """Return if the agent is at the decoration."""
        del obs, action

        if np.any(np.all(next_obs[:2] == self.DECORATION_COORD_LIST, axis=1)):
            return Symbol.D
        return None
