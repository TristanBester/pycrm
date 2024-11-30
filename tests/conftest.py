from enum import Enum, auto

import pytest


class EnvProps(Enum):
    """Enum modelling high-level environment events."""

    EVENT_A = auto()
    EVENT_B = auto()


@pytest.fixture
def env_props() -> type[EnvProps]:
    """Fixture for high-level environment events."""
    return EnvProps
