import os

# jax-metal crashes `import stoa` at module load. Force the CPU backend for the
# whole test process before JAX is first imported.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from enum import Enum, auto


class EnvProps(Enum):
    """Enum modelling high-level environment events."""

    EVENT_A = auto()
    EVENT_B = auto()
