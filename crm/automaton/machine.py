from abc import ABC, abstractmethod
from enum import Enum, EnumMeta
from typing import Any, Callable

import numpy as np

from crm.automaton.compiler import compile_transition_expression


class CountingRewardMachine(ABC):
    """Base class for all counting reward machines."""

    def __init__(self, env_prop_enum: EnumMeta) -> None:
        """Initialise the counting reward machine.

        Args:
            env_prop_enum (EnumMeta): Enum class containing environment properties.
        """
        self.env_prop_enum = env_prop_enum

        self._delta_u = self._get_state_transition_function()
        self._delta_c = self._get_counter_transition_function()
        self._delta_r = self._get_reward_transition_function()

        # Handle state-transition function
        self._replace_terminal_state()
        self.U = list(self._delta_u.keys())
        self.F = [self._get_max_state() + 1]

        # Handle reward-transition function
        self._replace_ccrm_rewards()
        self._init_transition_functions()

    @property
    @abstractmethod
    def u_0(self) -> int:
        """Return the initial state of the machine."""

    @property
    @abstractmethod
    def c_0(self) -> tuple[int, ...]:
        """Return the initial counter configuration of the machine."""

    @property
    @abstractmethod
    def encoded_configuration_size(self) -> int:
        """Return the size of the encoded counter configuration."""

    @abstractmethod
    def _get_state_transition_function(self) -> dict:
        """Return the state transition function."""

    @abstractmethod
    def _get_counter_transition_function(self) -> dict:
        """Return the counter transition function."""

    @abstractmethod
    def _get_reward_transition_function(self) -> dict:
        """Return the reward transition function."""

    @abstractmethod
    def sample_counter_configurations(self) -> list[tuple[int, ...]]:
        """Return counter configurations for counterfactual experience generation."""

    def encode_machine_state(self, u: int) -> np.ndarray:
        """Encode the machine state into a one-hot vector."""
        u_enc = np.zeros(self._get_max_state() + 2)
        u_enc[u] = 1
        return u_enc

    def encode_counter_configuration(
        self, c: tuple[int, ...], scale: float = 1
    ) -> np.ndarray:
        """Encode the counters as an array and rescale."""
        c_vec = np.array(c) / scale
        return c_vec

    def encode_counter_state(self, c: tuple[int, ...]) -> np.ndarray:
        """Encode the counter configuration as a one-hot vector."""
        c_enc = np.array([1 if c_i > 0 else 0 for c_i in c])
        return c_enc

    def transition(
        self, u: int, c: tuple[int, ...], props: set[Enum]
    ) -> tuple[int, tuple[int], Callable]:
        """Return the next state, counter configuration and reward function.

        Note: Transitions are applied in the order they are defined, so if multiple
        transitions are possible, the first one will be applied.

        Args:
            u (int): Current state.
            c (tuple[int]): Current counter configuration.
            props (set[EnumMeta]): Environment propositions.

        Returns:
            tuple[int, tuple[int], Callable]: Next state, counter configuration and
                reward function.
        """
        if u not in self.delta_u:
            raise ValueError(
                f"State u={u} is terminal or not defined in the transition function"
            )

        u_next = None
        c_next = None
        reward_fn = None

        for transition_formula in self.delta_u[u].keys():
            if transition_formula(props, c):
                u_next = self.delta_u[u][transition_formula]
                c_delta = self.delta_c[u][transition_formula]
                c_next = tuple(np.array(c) + np.array(c_delta))
                reward_fn = self.delta_r[u][transition_formula]
                break

        if u_next is not None and c_next is not None and reward_fn is not None:
            return u_next, c_next, reward_fn

        raise ValueError(
            f"Transition not defined for machine configuration ({u}, {c}) "
            f"and environment propositions {props}"
        )

    def _replace_terminal_state(self) -> None:
        """Replace the terminal state flag values in the state-transition function.

        Terminal states can be defined with index -1 when initialising
        the state-transition function. This function replaces these states
        with a new state which has the largest index in the state-transition
        function.
        """
        terminal_state = self._get_max_state() + 1

        for u in self._delta_u:
            for expr in self._delta_u[u]:
                if self._delta_u[u][expr] == -1:
                    self._delta_u[u][expr] = terminal_state

    def _replace_ccrm_rewards(self) -> None:
        """Convert constant counting reward machine to counting reward machine.

        Overwrite reward-transition function to return reward functions rather than
        scalar values.
        """
        for u in self._delta_r:
            for expr in self._delta_r[u]:
                if isinstance(self._delta_r[u][expr], float) or isinstance(
                    self._delta_r[u][expr], int
                ):
                    self._delta_r[u][expr] = self._create_constant_reward_function(
                        self._delta_r[u][expr]
                    )

    def _create_constant_reward_function(
        self, constant: float | int
    ) -> Callable[[Any, Any, Any], float]:
        """Create a constant reward function."""

        def constant_reward_function(obs, action, next_obs) -> float:
            del obs, action, next_obs
            return float(constant)

        return constant_reward_function

    def _init_transition_functions(self):
        """Initialise the transition functions."""
        self.delta_u = {}
        self.delta_c = {}
        self.delta_r = {}

        for u in self._delta_u.keys():
            d_u = {}
            d_c = {}
            d_r = {}

            for expr in self._delta_u[u]:
                transition_formula = compile_transition_expression(
                    expr, self.env_prop_enum
                )
                d_u[transition_formula] = self._delta_u[u][expr]

                try:
                    d_c[transition_formula] = self._delta_c[u][expr]
                except KeyError:
                    raise ValueError(
                        f"Missing counter configuration for transition {u}: {expr}"
                    ) from None

                try:
                    d_r[transition_formula] = self._delta_r[u][expr]
                except KeyError:
                    raise ValueError(
                        f"Missing reward function for transition {u}: {expr}"
                    ) from None

                reward_fn = self._delta_r[u][expr]
                d_r[transition_formula] = reward_fn

            self.delta_u[u] = d_u
            self.delta_c[u] = d_c
            self.delta_r[u] = d_r

    def _get_max_state(self) -> int:
        """Return the maximum state in the state-transition function."""
        max_state = 0
        for u in self._delta_u.keys():
            if u > max_state:
                max_state = u
        return max_state
