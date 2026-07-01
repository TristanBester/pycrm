"""Pure, jittable labelling functions for the JAX cross-product."""

import abc
from typing import Any


class JaxLabellingFunction(abc.ABC):
    """Base class for JAX-native labelling functions.

    A JAX labelling function maps a ground environment transition to a boolean
    vector of active propositions. Unlike ``pycrm.label.LabellingFunction`` (which
    returns a Python ``set`` of ``Enum`` events), this variant must be a pure,
    jittable function returning a fixed-size ``(num_props,)`` boolean array whose
    entries are ordered to match the machine's ``env_prop_enum``.
    """

    @abc.abstractmethod
    def __call__(
        self, ground_obs: Any, action: Any, next_ground_obs: Any
    ) -> Any:
        """Return a ``(num_props,)`` boolean array of active propositions.

        Args:
            ground_obs: The ground observation before the transition.
            action: The action taken.
            next_ground_obs: The ground observation after the transition.

        Returns:
            A ``jax.Array`` of dtype ``bool`` with one entry per proposition,
            ordered to match ``machine.env_prop_enum``.
        """
        raise NotImplementedError
