from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import Any, cast

import numpy as np

from pycrm.automaton import CountingRewardMachine, RewardMachine, RmToCrmAdapter
from pycrm.automaton.compiler import compile_transition_expression


@dataclass(frozen=True)
class JaxCompiledCRM:
    """Dense table representation of a counting reward machine."""

    u_0: int
    c_0: np.ndarray
    terminal_mask: np.ndarray
    next_state: np.ndarray
    counter_delta: np.ndarray
    reward: np.ndarray
    valid: np.ndarray
    counterfactual_machine_states: np.ndarray
    counterfactual_counter_configurations: np.ndarray
    num_props: int
    num_counters: int
    num_machine_states: int
    env_props: tuple[Enum, ...]


def compile_crm(crm: CountingRewardMachine | RewardMachine) -> JaxCompiledCRM:
    """Compile a counting reward machine into JAX-friendly dense tables.

    The existing CRM transition expressions are evaluated once for every
    proposition bitmask and counter zero/non-zero mask. At runtime, a JAX step
    only needs array indexing and counter arithmetic.
    """
    if not isinstance(crm, CountingRewardMachine):
        crm = RmToCrmAdapter(crm)

    env_props = cast(tuple[Enum, ...], tuple(crm.env_prop_enum))
    num_props = len(env_props)
    num_counters = len(crm.c_0)
    num_prop_masks = 1 << num_props
    num_counter_masks = 1 << num_counters
    num_machine_states = max([crm.u_0, *crm.U, *crm.F]) + 1

    table_shape = (num_machine_states, num_prop_masks, num_counter_masks)
    next_state = np.zeros(table_shape, dtype=np.int32)
    counter_delta = np.zeros((*table_shape, num_counters), dtype=np.int32)
    reward = np.zeros(table_shape, dtype=np.float32)
    valid = np.zeros(table_shape, dtype=np.bool_)
    terminal_mask = np.zeros(num_machine_states, dtype=np.bool_)

    for u in range(num_machine_states):
        next_state[u, :, :] = u

    for u in crm.F:
        terminal_mask[u] = True

    transition_formulas = _compile_transition_formulas(crm)

    for u in crm.U:
        for prop_mask in range(num_prop_masks):
            props = _props_from_mask(env_props, prop_mask)

            for counter_mask in range(num_counter_masks):
                counter_state = _counter_state_from_mask(num_counters, counter_mask)
                selected = _select_transition(
                    crm=crm,
                    transition_formulas=transition_formulas,
                    u=u,
                    props=props,
                    counter_state=counter_state,
                )

                if selected is None:
                    continue

                u_next, c_delta, scalar_reward = selected
                if len(c_delta) != num_counters:
                    raise ValueError(
                        "Counter delta for state "
                        + f"{u} has length {len(c_delta)}; expected {num_counters}."
                    )

                next_state[u, prop_mask, counter_mask] = u_next
                counter_delta[u, prop_mask, counter_mask] = np.array(
                    c_delta, dtype=np.int32
                )
                reward[u, prop_mask, counter_mask] = scalar_reward
                valid[u, prop_mask, counter_mask] = True

    return JaxCompiledCRM(
        u_0=int(crm.u_0),
        c_0=np.array(crm.c_0, dtype=np.int32),
        terminal_mask=terminal_mask,
        next_state=next_state,
        counter_delta=counter_delta,
        reward=reward,
        valid=valid,
        counterfactual_machine_states=np.array(crm.U, dtype=np.int32),
        counterfactual_counter_configurations=np.array(
            crm.sample_counter_configurations(), dtype=np.int32
        ),
        num_props=num_props,
        num_counters=num_counters,
        num_machine_states=num_machine_states,
        env_props=env_props,
    )


def _compile_transition_formulas(crm: CountingRewardMachine) -> dict[int, list[tuple]]:
    transition_formulas = {}

    for u, transitions in crm._delta_u.items():
        transition_formulas[u] = [
            (expr, compile_transition_expression(expr, crm.env_prop_enum))
            for expr in transitions
        ]

    return transition_formulas


def _select_transition(
    *,
    crm: CountingRewardMachine,
    transition_formulas: dict[int, list[tuple[str, Any]]],
    u: int,
    props: set[Enum],
    counter_state: tuple[int, ...],
) -> tuple[int, tuple[int, ...], float] | None:
    for expr, transition_formula in transition_formulas[u]:
        if not transition_formula(props, counter_state):
            continue

        return (
            int(crm._delta_u[u][expr]),
            tuple(int(delta) for delta in crm._delta_c[u][expr]),
            _scalar_reward_value(crm._delta_r[u][expr], u, expr),
        )

    return None


def _scalar_reward_value(reward_fn: Any, u: int, expr: str) -> float:
    if isinstance(reward_fn, Real):
        return float(reward_fn)

    reward_value = getattr(reward_fn, "_pycrm_constant_reward", None)
    if reward_value is not None:
        return float(reward_value)

    raise TypeError(
        "JAX CRM compilation only supports scalar reward transitions. "
        + f"Transition {u}: {expr!r} has a Python reward callable."
    )


def _props_from_mask(env_props: tuple[Enum, ...], prop_mask: int) -> set[Enum]:
    return {
        prop for idx, prop in enumerate(env_props) if prop_mask & (1 << idx)
    }


def _counter_state_from_mask(
    num_counters: int, counter_mask: int
) -> tuple[int, ...]:
    return tuple(1 if counter_mask & (1 << idx) else 0 for idx in range(num_counters))
