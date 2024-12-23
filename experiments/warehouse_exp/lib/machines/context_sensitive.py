from itertools import product
from typing import Callable

import numpy as np

import experiments.warehouse_exp.lib.constants.waypoints as cw
from crm.automaton import CountingRewardMachine
from crm.automaton.compiler import compile_transition_expression
from experiments.warehouse_exp.lib.label.events import PickPlaceEvent


class ContextSensitiveCRM(CountingRewardMachine):
    def __init__(self, c_0: tuple[int, int, int] = (3, 3, 3)) -> None:
        super().__init__(env_prop_enum=PickPlaceEvent)
        self._c_0 = c_0
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

    # OVERRIDE
    def transition(
        self, u: int, c: tuple[int], props
    ) -> tuple[int, tuple[int], Callable]:
        """
        NOTE: Transitions are applied in the order they are defined, so if multiple transitions are possible, the first one will be applied.
        """
        if u not in self.delta_u:
            print(u, c, props)

            raise ValueError(
                f"State u={u} is terminal or not defined in the transition function"
            )

        u_next = (None,)
        c_next = None
        reward_fn = None

        for transition_formula in self.delta_u[u].keys():
            if transition_formula(props, c):
                u_next = self.delta_u[u][transition_formula]
                c_delta = self.delta_c[u][transition_formula]
                c_next = tuple(np.array(c) + np.array(c_delta))
                reward_fn = self.delta_r[u][transition_formula]
                break

        if u_next is None or c_next is None or reward_fn is None:
            raise ValueError(
                f"Transition not defined for machine configuration ({u}, {c}) and environment propositions {props}"
            )

        # TODO: Fix typing
        return u_next, c_next, reward_fn  # type: ignore

    @property
    def u_0(self) -> int:
        return 0

    @property
    def c_0(self) -> tuple[int, int, int]:
        return self._c_0

    def _get_state_transition_function(self) -> dict:
        self._delta_u = {
            0: {
                # All blocks packed
                "/ (Z, Z, Z)": -1,
                # Pack green blocks
                "/ (NZ, -, -)": 15,
                # Pack red blocks
                "/ (Z, NZ, -)": 10,
                # Pack blue blocks
                "/ (Z, Z, NZ)": 5,
            },
        }

        # Add blue block module
        self._add_block_module_delta_u(
            u_start=1, counter_state="(Z, Z, NZ)", block_colour="BLUE"
        )
        # Add red block module
        self._add_block_module_delta_u(
            u_start=6, counter_state="(Z, NZ, -)", block_colour="GREEN"
        )
        # Add green block module
        self._add_block_module_delta_u(
            u_start=11, counter_state="(NZ, -, -)", block_colour="RED"
        )
        return self._delta_u

    def _get_counter_transition_function(self) -> dict:
        self._delta_c = {
            0: {
                "/ (Z, Z, Z)": (0, 0, 0),
                "/ (NZ, -, -)": (0, 0, 0),
                "/ (Z, NZ, -)": (0, 0, 0),
                "/ (Z, Z, NZ)": (0, 0, 0),
            },
        }

        # Add blue block module
        self._add_block_module_delta_c(
            u_start=1, counter_state="(Z, Z, NZ)", block_colour="BLUE"
        )
        # Add red block module
        self._add_block_module_delta_c(
            u_start=6, counter_state="(Z, NZ, -)", block_colour="GREEN"
        )
        # Add green block module
        self._add_block_module_delta_c(
            u_start=11, counter_state="(NZ, -, -)", block_colour="RED"
        )
        return self._delta_c

    def _get_reward_transition_function(self) -> dict:
        self._delta_r = {
            0: {
                "/ (Z, Z, Z)": self._create_f_c(0.0),
                "/ (NZ, -, -)": self._create_f_c(0.0),
                "/ (Z, NZ, -)": self._create_f_c(0.0),
                "/ (Z, Z, NZ)": self._create_f_c(0.0),
            },
        }

        # Add blue block module
        self._add_block_module_delta_r(
            u_start=1, counter_state="(Z, Z, NZ)", block_colour="BLUE"
        )
        # Add red block module
        self._add_block_module_delta_r(
            u_start=6, counter_state="(Z, NZ, -)", block_colour="GREEN"
        )
        # Add green block module
        self._add_block_module_delta_r(
            u_start=11, counter_state="(NZ, -, -)", block_colour="RED"
        )
        return self._delta_r

    def sample_counter_configurations(self) -> list[tuple[int, ...]]:
        main_configurations = list(product(range(3), repeat=3))

        random_configurations = np.random.randint(low=0, high=100_000, size=(15, 3))
        random_configurations[5:, 0] = 0
        random_configurations[10:, 1] = 0
        random_configurations = [tuple(config) for config in random_configurations]

        configurations = main_configurations + random_configurations
        return configurations

    def _add_block_module_delta_u(
        self, u_start: int, counter_state: str, block_colour: str
    ):
        module_terminal_state = 0
        base_u = u_start + 4

        delta_u = {
            base_u: {
                f"not GRIPPER_OPEN_ACTION_EXECUTED / {counter_state}": base_u,
                f"ABOVE_{block_colour} and VELOCITY_LOW / {counter_state}": base_u - 1,
                f"not (ABOVE_{block_colour} and VELOCITY_LOW) / {counter_state}": base_u,
            },
            base_u - 1: {
                f"not GRIPPER_OPEN_ACTION_EXECUTED / {counter_state}": (base_u - 1),
                f"not SAFE_REGION_{block_colour} / {counter_state}": (base_u - 1),
                f"GRASP_{block_colour} and VELOCITY_LOW / {counter_state}": (base_u - 1)
                - 1,
                f"not (GRASP_{block_colour} and VELOCITY_LOW) / {counter_state}": (
                    base_u - 1
                ),
            },
            base_u - 2: {
                f"GRIPPER_OPEN_ACTION_EXECUTED / {counter_state}": (base_u - 2),
                f"not TIGHT_REGION_{block_colour} / {counter_state}": (base_u - 2),
                f"TIGHT_REGION_{block_colour} and GRIPPER_CLOSED / {counter_state}": (
                    base_u - 2
                )
                - 1,
                f"TIGHT_REGION_{block_colour} and not GRIPPER_CLOSED / {counter_state}": (
                    base_u - 2
                ),
            },
            base_u - 3: {
                f"GRIPPER_OPEN_ACTION_EXECUTED / {counter_state}": (base_u - 3),
                f"not RELEASE_REGION_{block_colour} / {counter_state}": (base_u - 3),
                f"RELEASE_{block_colour} and VELOCITY_LOW / {counter_state}": (
                    base_u - 3
                )
                - 1,
                f"not (RELEASE_{block_colour} and VELOCITY_LOW) / {counter_state}": (
                    base_u - 3
                ),
            },
            base_u - 4: {
                f"not GRIPPER_OPEN_ACTION_EXECUTED / {counter_state}": (base_u - 4),
                f"not RELEASE_REGION_{block_colour} / {counter_state}": (base_u - 4),
                f"RELEASE_REGION_{block_colour} and not GRIPPER_CLOSED / {counter_state}": module_terminal_state,
                f"RELEASE_REGION_{block_colour} and GRIPPER_CLOSED / {counter_state}": (
                    base_u - 4
                ),
            },
        }

        # Merge delta_u
        for u in delta_u.keys():
            if u not in self._delta_u.keys():
                self._delta_u[u] = {}
            self._delta_u[u] |= delta_u[u]

    def _add_block_module_delta_c(
        self, u_start: int, counter_state: str, block_colour: str
    ):
        if block_colour == "RED":
            module_terminal_counter_modifier = (-1, 0, 0)
        elif block_colour == "GREEN":
            module_terminal_counter_modifier = (0, -1, 0)
        elif block_colour == "BLUE":
            module_terminal_counter_modifier = (0, 0, -1)
        else:
            raise ValueError(f"Invalid block colour: {block_colour}")

        base_u = u_start + 4

        delta_c = {
            base_u: {
                f"not GRIPPER_OPEN_ACTION_EXECUTED / {counter_state}": (0,),
                f"ABOVE_{block_colour} and VELOCITY_LOW / {counter_state}": (0,),
                f"not (ABOVE_{block_colour} and VELOCITY_LOW) / {counter_state}": (0,),
            },
            base_u - 1: {
                f"not GRIPPER_OPEN_ACTION_EXECUTED / {counter_state}": (0,),
                f"not SAFE_REGION_{block_colour} / {counter_state}": (0,),
                f"GRASP_{block_colour} and VELOCITY_LOW / {counter_state}": (0,),
                f"not (GRASP_{block_colour} and VELOCITY_LOW) / {counter_state}": (0,),
            },
            base_u - 2: {
                f"GRIPPER_OPEN_ACTION_EXECUTED / {counter_state}": (0,),
                f"not TIGHT_REGION_{block_colour} / {counter_state}": (0,),
                f"TIGHT_REGION_{block_colour} and GRIPPER_CLOSED / {counter_state}": (
                    0,
                ),
                f"TIGHT_REGION_{block_colour} and not GRIPPER_CLOSED / {counter_state}": (
                    0,
                ),
            },
            base_u - 3: {
                f"GRIPPER_OPEN_ACTION_EXECUTED / {counter_state}": (0,),
                f"not RELEASE_REGION_{block_colour} / {counter_state}": (0,),
                f"RELEASE_{block_colour} and VELOCITY_LOW / {counter_state}": (0,),
                f"not (RELEASE_{block_colour} and VELOCITY_LOW) / {counter_state}": (
                    0,
                ),
            },
            base_u - 4: {
                f"not GRIPPER_OPEN_ACTION_EXECUTED / {counter_state}": (0,),
                f"not RELEASE_REGION_{block_colour} / {counter_state}": (0,),
                f"RELEASE_REGION_{block_colour} and not GRIPPER_CLOSED / {counter_state}": module_terminal_counter_modifier,
                f"RELEASE_REGION_{block_colour} and GRIPPER_CLOSED / {counter_state}": (
                    0,
                ),
            },
        }

        # Merge delta_c
        for u in delta_c.keys():
            if u not in self._delta_c.keys():
                self._delta_c[u] = {}
            self._delta_c[u] |= delta_c[u]

    def _add_block_module_delta_r(
        self, u_start: int, counter_state: str, block_colour: str
    ):
        if block_colour == "GREEN":
            release_waypoint = cw.RELEASE_GREEN
        elif block_colour == "RED":
            release_waypoint = cw.RELEASE_RED
        elif block_colour == "BLUE":
            release_waypoint = cw.RELEASE_BLUE
        else:
            raise ValueError(f"Invalid block colour: {block_colour}")

        waypoint_above = getattr(cw, f"ABOVE_{block_colour}")
        waypoint_grasp = getattr(cw, f"GRASP_{block_colour}")

        base_u = u_start + 4

        delta_r = {
            base_u: {
                f"not GRIPPER_OPEN_ACTION_EXECUTED / {counter_state}": self._create_unscaled_f_c(
                    -2
                ),
                f"ABOVE_{block_colour} and VELOCITY_LOW / {counter_state}": self._create_f_c(
                    500
                ),
                f"not (ABOVE_{block_colour} and VELOCITY_LOW) / {counter_state}": self._create_f_w(
                    waypoint_above
                ),
            },
            base_u - 1: {
                f"not GRIPPER_OPEN_ACTION_EXECUTED / {counter_state}": self._create_unscaled_f_c(
                    -2
                ),
                f"not SAFE_REGION_{block_colour} / {counter_state}": self._create_f_pw(
                    waypoint_grasp,
                ),
                f"GRASP_{block_colour} and VELOCITY_LOW / {counter_state}": self._create_f_c(
                    500
                ),
                f"not (GRASP_{block_colour} and VELOCITY_LOW) / {counter_state}": self._create_f_w(
                    waypoint_grasp,
                ),
            },
            base_u - 2: {
                f"GRIPPER_OPEN_ACTION_EXECUTED / {counter_state}": self._create_unscaled_f_c(
                    -2
                ),
                f"not TIGHT_REGION_{block_colour} / {counter_state}": self._create_f_pw(
                    waypoint_grasp,
                ),
                f"TIGHT_REGION_{block_colour} and GRIPPER_CLOSED / {counter_state}": self._create_f_c(
                    500
                ),
                f"TIGHT_REGION_{block_colour} and not GRIPPER_CLOSED / {counter_state}": self._create_f_w(
                    waypoint_grasp,
                ),
            },
            base_u - 3: {
                f"GRIPPER_OPEN_ACTION_EXECUTED / {counter_state}": self._create_unscaled_f_c(
                    -2
                ),
                f"not RELEASE_REGION_{block_colour} / {counter_state}": self._create_f_pw(
                    release_waypoint
                ),
                f"RELEASE_{block_colour} and VELOCITY_LOW / {counter_state}": self._create_f_c(
                    500
                ),
                f"not (RELEASE_{block_colour} and VELOCITY_LOW) / {counter_state}": self._create_f_w(
                    release_waypoint
                ),
            },
            base_u - 4: {
                f"not GRIPPER_OPEN_ACTION_EXECUTED / {counter_state}": self._create_unscaled_f_c(
                    -2
                ),
                f"not RELEASE_REGION_{block_colour} / {counter_state}": self._create_f_pw(
                    release_waypoint
                ),
                f"RELEASE_REGION_{block_colour} and not GRIPPER_CLOSED / {counter_state}": self._create_unscaled_f_c(
                    500
                ),
                f"RELEASE_REGION_{block_colour} and GRIPPER_CLOSED / {counter_state}": self._create_f_w(
                    release_waypoint
                ),
            },
        }

        for u in delta_r.keys():
            if u not in self._delta_r.keys():
                self._delta_r[u] = {}
            self._delta_r[u] |= delta_r[u]

    # NOTE: Custom override of _init_transition_functions
    def _init_transition_functions(self):
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
                    )

                if "ACTION" not in expr:
                    reward_fn = self._delta_r[u][expr]
                    if u != 0:
                        scaled_reward_fn = self._create_scaled_reward_function(
                            reward_fn, u, 15
                        )
                    else:
                        u_next = self._delta_u[u][expr]

                        if u_next != 15:
                            scaled_reward_fn = self._create_scaled_reward_function(
                                reward_fn, u_next, 15
                            )
                        else:
                            scaled_reward_fn = reward_fn
                else:
                    reward_fn = self._delta_r[u][expr]
                    scaled_reward_fn = reward_fn

                d_r[transition_formula] = scaled_reward_fn
            self.delta_u[u] = d_u
            self.delta_c[u] = d_c
            self.delta_r[u] = d_r

    def _create_scaled_reward_function(self, reward_fn: Callable, u: int, max_u: int):
        def scaled_reward_fn(
            obs: np.ndarray,
            action: np.ndarray,
            obs_next: np.ndarray,
        ) -> float:
            reward = reward_fn(obs, action, obs_next)
            modified_reward = reward - u
            scaled_reward = ((modified_reward - max_u) / (max_u)) + 2  # In [0, 1]
            return scaled_reward - 1

        return scaled_reward_fn

    def _create_f_c(self, reward_val: float):
        """Return a constant real number."""

        def f_c(
            obs: dict[str, np.ndarray],
            action: np.ndarray,
            obs_next: dict[str, np.ndarray],
        ) -> float:
            del obs, action, obs_next
            return np.clip(reward_val, 0.0, 1.0)  # In [0, 1]

        return f_c

    def _create_unscaled_f_c(self, reward_val: float):
        """Return a constant real number."""

        def f_c(
            obs: dict[str, np.ndarray],
            action: np.ndarray,
            obs_next: dict[str, np.ndarray],
        ) -> float:
            del obs, action, obs_next
            return reward_val

        return f_c

    def _create_f_w(self, waypoint_postion: np.ndarray):
        """Reward agent for moving towards waypoint."""

        def f_w(
            obs: np.ndarray,
            action: np.ndarray,
            obs_next: np.ndarray,
        ) -> float:
            del action, obs
            ee_pos_next = obs_next[:3]
            pos_err_next = float(np.linalg.norm(ee_pos_next - waypoint_postion))
            unscaled_reward = -1.0 * pos_err_next  # In [-inf, 0]
            clipped_reward = np.clip(unscaled_reward, -0.25, 0.0)  # In [-1, 0]
            scaled_reward = (clipped_reward - (-0.25)) / (0.0 - (-0.25))  # In [0, 1]
            return scaled_reward

        return f_w

    def _create_f_pw(self, waypoint: np.ndarray):
        """Penalise agent for being outside of safe region."""
        f_w = self._create_f_w(waypoint)

        def f_pw(
            obs: np.ndarray,
            action: np.ndarray,
            obs_next: np.ndarray,
        ) -> float:
            pos_reward = f_w(obs, action, obs_next)
            return -1 + pos_reward

        return f_pw


if __name__ == "__main__":
    crm = ContextSensitiveCRM()
