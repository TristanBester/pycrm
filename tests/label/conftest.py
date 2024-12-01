from enum import Enum

import numpy as np
import pytest

from crm.label import LabellingFunction
from tests.conftest import EnvProps


class AllEventsLabellingFunction(LabellingFunction[np.ndarray, np.ndarray]):
    """Labelling function that returns all events."""

    @LabellingFunction.event
    def test_event_a(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> Enum | None:
        """Return event A."""
        return EnvProps.EVENT_A

    @LabellingFunction.event
    def test_event_b(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> Enum:
        """Return event B."""
        return EnvProps.EVENT_B


class NoEventsLabellingFunction(LabellingFunction[np.ndarray, np.ndarray]):
    """Labelling function that returns no events."""

    @LabellingFunction.event
    def test_event_a(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> None:
        """Return event A."""
        return None

    @LabellingFunction.event
    def test_event_b(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> None:
        """Return event B."""
        return None


class OneEventLabellingFunction(LabellingFunction[np.ndarray, np.ndarray]):
    """Labelling function that returns one event."""

    @LabellingFunction.event
    def test_event_a(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> Enum:
        """Return event A."""
        return EnvProps.EVENT_A

    @LabellingFunction.event
    def test_event_b(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> None:
        """Return event B."""
        return None


@pytest.fixture
def all_events_labelling_function() -> AllEventsLabellingFunction:
    """Fixture for labelling function that returns all events."""
    return AllEventsLabellingFunction()


@pytest.fixture
def no_events_labelling_function() -> NoEventsLabellingFunction:
    """Fixture for labelling function that returns no events."""
    return NoEventsLabellingFunction()


@pytest.fixture
def one_event_labelling_function() -> OneEventLabellingFunction:
    """Fixture for labelling function that returns one event."""
    return OneEventLabellingFunction()
