from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import Any, cast

import numpy as np

from pycrm.automaton import CountingRewardMachine, RewardMachine, RmToCrmAdapter
from pycrm.automaton.compiler import compile_transition_expression

#: Attribute set by :func:`jax_reward` marking a reward callable as safe to
#: trace inside a JAX transform. Compilation registers marked callables for
#: runtime dispatch instead of rejecting them.
JAX_REWARD_ATTR = "_pycrm_jax_reward"


def jax_reward(reward_fn: Callable) -> Callable:
    """Mark a CRM reward callable as JAX-traceable for dynamic-reward dispatch.

    Reward transitions in a :class:`CountingRewardMachine` may be plain scalars
    or Python callables ``(obs, action, next_obs) -> reward``. Scalars are baked
    into the dense reward table at compile time. A callable is normally rejected
    by :func:`compile_crm` because arbitrary Python cannot be traced by JAX.

    Decorating the callable with ``@jax_reward`` is a promise that its body is
    pure and JAX-traceable (operating on ``jax.Array`` inputs, returning a
    scalar, no Python control flow on traced values). Marked callables are
    collected into a per-machine registry and dispatched at runtime via
    ``jax.lax.switch``, so the machine definition remains the single source of
    truth for rewards in both the NumPy and JAX runtimes.

    Args:
        reward_fn: A pure ``(obs, action, next_obs) -> reward`` callable.

    Returns:
        The same callable, tagged so :func:`compile_crm` registers it.
    """
    setattr(reward_fn, JAX_REWARD_ATTR, True)
    return reward_fn


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
    #: ``[u, prop_mask, counter_mask]`` int32 table. ``0`` means "use the scalar
    #: ``reward`` table"; ``k > 0`` selects ``reward_fns[k - 1]`` at runtime.
    reward_fn_id: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0, 0), dtype=np.int32)
    )
    #: Registry of ``@jax_reward``-marked callables, indexed by ``id - 1``.
    reward_fns: tuple[Callable, ...] = ()


def compile_crm(
    crm: CountingRewardMachine | RewardMachine,
    *,
    allow_dynamic_rewards: bool = False,
) -> JaxCompiledCRM:
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
    reward_fn_id = np.zeros(table_shape, dtype=np.int32)
    valid = np.zeros(table_shape, dtype=np.bool_)
    terminal_mask = np.zeros(num_machine_states, dtype=np.bool_)

    reward_registry = _RewardRegistry()

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
                    allow_dynamic_rewards=allow_dynamic_rewards,
                )

                if selected is None:
                    continue

                u_next, c_delta, scalar_reward, dynamic_reward_fn = selected
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
                if dynamic_reward_fn is not None:
                    reward_fn_id[u, prop_mask, counter_mask] = reward_registry.id_of(
                        dynamic_reward_fn
                    )
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
        reward_fn_id=reward_fn_id,
        reward_fns=reward_registry.as_tuple(),
    )


def _compile_transition_formulas(crm: CountingRewardMachine) -> dict[int, list[tuple]]:
    transition_formulas = {}

    for u, transitions in crm._delta_u.items():
        transition_formulas[u] = [
            (expr, compile_transition_expression(expr, crm.env_prop_enum))
            for expr in transitions
        ]

    return transition_formulas


class _RewardRegistry:
    """Deduplicating registry assigning 1-based ids to reward callables."""

    def __init__(self) -> None:
        self._ids: dict[int, int] = {}
        self._fns: list[Callable] = []

    def id_of(self, reward_fn: Callable) -> int:
        """Return the 1-based id for ``reward_fn``, registering it if new."""
        key = id(reward_fn)
        existing = self._ids.get(key)
        if existing is not None:
            return existing

        new_id = len(self._fns) + 1
        self._ids[key] = new_id
        self._fns.append(reward_fn)
        return new_id

    def as_tuple(self) -> tuple[Callable, ...]:
        """Return the registered callables ordered by ``id - 1``."""
        return tuple(self._fns)


def _select_transition(
    *,
    crm: CountingRewardMachine,
    transition_formulas: dict[int, list[tuple[str, Any]]],
    u: int,
    props: set[Enum],
    counter_state: tuple[int, ...],
    allow_dynamic_rewards: bool,
) -> tuple[int, tuple[int, ...], float, Callable | None] | None:
    for expr, transition_formula in transition_formulas[u]:
        if not transition_formula(props, counter_state):
            continue

        scalar_reward, dynamic_reward_fn = _classify_reward(
            crm._delta_r[u][expr],
            u,
            expr,
            allow_dynamic_rewards=allow_dynamic_rewards,
        )
        return (
            int(crm._delta_u[u][expr]),
            tuple(int(delta) for delta in crm._delta_c[u][expr]),
            scalar_reward,
            dynamic_reward_fn,
        )

    return None


def _classify_reward(
    reward_fn: Any,
    u: int,
    expr: str,
    *,
    allow_dynamic_rewards: bool,
) -> tuple[float, Callable | None]:
    """Resolve a CRM reward into a ``(scalar, dynamic_callable)`` pair.

    Exactly one side of the pair is meaningful per transition: a scalar reward
    yields ``(value, None)`` while a ``@jax_reward``-marked callable yields
    ``(0.0, callable)`` for runtime ``lax.switch`` dispatch.
    """
    if isinstance(reward_fn, Real):
        return float(reward_fn), None

    reward_value = getattr(reward_fn, "_pycrm_constant_reward", None)
    if reward_value is not None:
        return float(reward_value), None

    if getattr(reward_fn, JAX_REWARD_ATTR, False):
        return 0.0, reward_fn

    if allow_dynamic_rewards:
        # Legacy path: the reward is supplied at runtime via a ``reward_fn``
        # override on the cross-product core, so the table slot stays zero.
        return 0.0, None

    raise TypeError(
        "JAX CRM compilation only supports scalar reward transitions. "
        + f"Transition {u}: {expr!r} has a Python reward callable. "
        + "Decorate it with @pycrm.jax.jax_reward to enable runtime dispatch, "
        + "or pass allow_dynamic_rewards=True to supply rewards via reward_fn."
    )


def _props_from_mask(env_props: tuple[Enum, ...], prop_mask: int) -> set[Enum]:
    return {prop for idx, prop in enumerate(env_props) if prop_mask & (1 << idx)}


def _counter_state_from_mask(num_counters: int, counter_mask: int) -> tuple[int, ...]:
    return tuple(1 if counter_mask & (1 << idx) else 0 for idx in range(num_counters))
