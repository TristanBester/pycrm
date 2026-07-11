import os
import time
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Callable

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp


class JaxLabellingFunction(ABC):
    _event_tests: tuple[Callable, ...] = ()

    @classmethod
    @abstractmethod
    def event_enum(cls) -> type[Enum]:
        """Return the enum that defines proposition-vector ordering."""
        raise NotImplementedError

    @staticmethod
    def event(proposition: Enum):
        def decorator(func: Callable):
            func._crm_proposition = proposition
            return func

        return decorator

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        # Check event enum not missing
        if getattr(cls.event_enum, "__isabstractmethod__", False):
            return

        propositions = tuple(cls.event_enum())
        if not propositions:
            raise ValueError("event_enum must not be empty")

        event_tests = {}

        for value in cls.__dict__.values():
            proposition = getattr(value, "_crm_proposition", None)
            if proposition is None:
                continue

            if not isinstance(proposition, cls.event_enum()):
                raise TypeError(
                    f"proposition {proposition} is not a valid member of event_enum"
                )

            if proposition in event_tests:
                raise ValueError(f"proposition {proposition} is defined multiple times")

            event_tests[proposition] = value

        missing = set(propositions) - set(event_tests)
        if missing:
            missing_names = ", ".join(proposition.name for proposition in missing)
            raise ValueError(
                f"Event detection functions are not defined for the following propositions: {missing_names}"
            )

        cls._event_tests = tuple(event_tests[prop] for prop in propositions)

    def __call__(self, obs, action, next_obs):
        truth_values = tuple(
            test(self, obs, action, next_obs) for test in self._event_tests
        )
        return jnp.array(truth_values, dtype=jnp.bool_)


class Symbol(Enum):
    A = auto()
    B = auto()


class GridLF(JaxLabellingFunction):
    symbols = Symbol

    def __init__(self, grid_size: int) -> None:
        self.a_pos = jnp.array([grid_size - 1, 0], dtype=jnp.int32)
        self.b_pos = jnp.array([grid_size - 1, grid_size - 1], dtype=jnp.int32)

    @classmethod
    def event_enum(cls) -> type[Enum]:
        return cls.symbols

    @JaxLabellingFunction.event(Symbol.A)
    def at_a(self, obs, action, next_obs):
        del obs, action
        return jnp.all(next_obs == self.a_pos)

    @JaxLabellingFunction.event(Symbol.B)
    def at_b(self, obs, action, next_obs):
        del obs, action
        return jnp.all(next_obs == self.b_pos)


if __name__ == "__main__":
    lf = GridLF(5)
    print(lf(jnp.array([0, 0]), 0, jnp.array([0, 0])))
    print(lf(jnp.array([0, 0]), 0, jnp.array([4, 0])))
    print(lf(jnp.array([0, 0]), 0, jnp.array([4, 4])))
