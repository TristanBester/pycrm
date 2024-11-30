import numpy as np

from crm.label import LabellingFunction
from tests.conftest import EnvProps


class TestLabellingFunction:
    """Test labelling function implementation."""

    def test_all_events(self, all_events_labelling_function: LabellingFunction) -> None:
        """Test all events returned."""
        events = all_events_labelling_function(
            obs=np.zeros(1),
            action=np.zeros(1),
            next_obs=np.zeros(1),
        )
        assert isinstance(events, set)
        assert len(events) == 2
        assert EnvProps.EVENT_A in events
        assert EnvProps.EVENT_B in events

    def test_no_events(self, no_events_labelling_function: LabellingFunction) -> None:
        """Test no events returned."""
        events = no_events_labelling_function(
            obs=np.zeros(1),
            action=np.zeros(1),
            next_obs=np.zeros(1),
        )
        assert isinstance(events, set)
        assert len(events) == 0

    def test_one_event(self, one_event_labelling_function: LabellingFunction) -> None:
        """Test one event returned."""
        events = one_event_labelling_function(
            obs=np.zeros(1), action=np.zeros(1), next_obs=np.zeros(1)
        )
        assert isinstance(events, set)
        assert len(events) == 1
        assert EnvProps.EVENT_A in events
        assert EnvProps.EVENT_B not in events
