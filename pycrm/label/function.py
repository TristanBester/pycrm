from abc import ABC
from enum import Enum
from typing import Any, Callable, Generic, TypeVar

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class LabellingFunction(ABC, Generic[ObsType, ActType]):
    """Base class for labelling functions.

    Labelling functions are mappings from environmental observations, to
    sets of high-level events taking place in the environment. In this context,
    events are modelled by an Enum.

    To implement a labelling function, create a class inheriting from
    `LabellingFunction` and decorate methods with the `event` decorator.
    The purpose of the decorated methods is to accept an environment transition
    and test whether a given event is taking place. If so, return the
    appropriate Enum. If not, return `None`.
    """

    @staticmethod
    def event(
        func: Callable[[Any, ObsType, ActType, ObsType], Enum | None],
    ) -> Callable[[Any, ObsType, ActType, ObsType], Enum | None]:
        """Register an event test."""

        def wrapper(self, *args, **kwargs) -> Enum | None:
            """The decorated method."""
            return func(self, *args, **kwargs)

        setattr(wrapper, "_is_event_method", True)
        return wrapper

    def __call__(self, obs: ObsType, action: ActType, next_obs: ObsType) -> set[Enum]:
        """Return set of high-level events taking place."""
        events = set()

        for method_name in dir(self):
            method = getattr(self, method_name)

            if callable(method) and getattr(method, "_is_event_method", False):
                result = method(obs, action, next_obs)
                if result is not None:
                    events.add(result)
        return events
