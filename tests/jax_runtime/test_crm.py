import numpy as np
import pytest

pytest.importorskip("jax.numpy")

from examples.introduction.core.label import Symbol  # noqa: E402
from examples.introduction.core.machine import (  # noqa: E402
    LetterWorldCountingRewardMachine,
)
from pycrm.automaton import CountingRewardMachine, RewardMachine  # noqa: E402
from pycrm.jax import compile_crm, jax_reward  # noqa: E402
from tests.crossproduct.conftest import CRM, Events  # noqa: E402


def test_compiled_crm_matches_python_transitions() -> None:
    """Compiled CRM tables match the existing Python transition function."""
    crm = CRM(env_prop_enum=Events)
    compiled = compile_crm(crm)

    for u in crm.U:
        for c in crm.sample_counter_configurations():
            for prop_mask in range(1 << compiled.num_props):
                props = _props_from_mask(compiled.env_props, prop_mask)
                counter_mask = _counter_mask(c)

                try:
                    u_next, c_next, reward_fn = crm.transition(u, c, props)
                except ValueError:
                    assert not compiled.valid[u, prop_mask, counter_mask]
                    continue

                assert compiled.valid[u, prop_mask, counter_mask]
                assert compiled.next_state[u, prop_mask, counter_mask] == u_next
                np.testing.assert_array_equal(
                    np.asarray(c) + compiled.counter_delta[u, prop_mask, counter_mask],
                    np.asarray(c_next),
                )
                assert compiled.reward[u, prop_mask, counter_mask] == reward_fn(
                    np.array([0]), 0, np.array([1])
                )


def test_compiled_crm_preserves_transition_priority() -> None:
    """Earlier matching CRM transitions win over later matching transitions."""
    crm = PriorityCRM(env_prop_enum=Events)
    compiled = compile_crm(crm)
    event_a_mask = 1 << list(compiled.env_props).index(Events.EVENT_A)

    assert compiled.next_state[0, event_a_mask, 0] == 1
    assert compiled.reward[0, event_a_mask, 0] == 1.0
    assert compiled.next_state[0, 0, 0] == 2
    assert compiled.reward[0, 0, 0] == 2.0


def test_compiled_crm_marks_terminal_replacement() -> None:
    """Terminal ``-1`` transitions are replaced before JAX compilation."""
    crm = LetterWorldCountingRewardMachine()
    compiled = compile_crm(crm)
    c_mask = 0
    c_prop_mask = 1 << list(compiled.env_props).index(Symbol.C)
    u_next = compiled.next_state[1, c_prop_mask, c_mask]

    assert u_next in crm.F
    assert compiled.terminal_mask[u_next]


def test_compiled_crm_rejects_python_reward_callables() -> None:
    """Unmarked Python reward callables are still rejected by compilation."""
    crm = DynamicRewardCRM(env_prop_enum=Events)

    with pytest.raises(TypeError, match="scalar reward transitions"):
        compile_crm(crm)


def test_compiled_crm_registers_jax_reward_callables() -> None:
    """@jax_reward callables are registered for runtime dispatch."""
    crm = JaxDynamicRewardCRM(env_prop_enum=Events)
    compiled = compile_crm(crm)
    event_a_mask = 1 << list(compiled.env_props).index(Events.EVENT_A)
    event_b_mask = 1 << list(compiled.env_props).index(Events.EVENT_B)

    # The scalar slot is zeroed; the id selects the registered callable.
    assert compiled.reward[0, event_a_mask, 0] == 0.0
    assert compiled.reward_fn_id[0, event_a_mask, 0] == 1
    assert compiled.reward_fns == (_jax_dynamic_reward,)

    # Scalar transitions keep id 0 (i.e. "use the reward table").
    assert compiled.reward_fn_id[0, event_b_mask, 0] == 0


def test_compiled_crm_deduplicates_shared_reward_callables() -> None:
    """The same callable reused across transitions gets a single id."""
    crm = SharedJaxRewardCRM(env_prop_enum=Events)
    compiled = compile_crm(crm)
    event_a_mask = 1 << list(compiled.env_props).index(Events.EVENT_A)

    assert compiled.reward_fns == (_jax_dynamic_reward,)
    assert compiled.reward_fn_id[0, event_a_mask, 0] == 1
    assert compiled.reward_fn_id[1, event_a_mask, 0] == 1


def test_compile_crm_accepts_reward_machine_adapter() -> None:
    """Plain reward machines compile through the same RM-to-CRM adapter path."""
    rm = TerminalRewardMachine(env_prop_enum=Events)
    compiled = compile_crm(rm)
    event_a_mask = 1 << list(compiled.env_props).index(Events.EVENT_A)
    u_next = compiled.next_state[0, event_a_mask, 0]

    assert compiled.num_counters == 1
    assert compiled.valid[0, event_a_mask, 0]
    assert compiled.reward[0, event_a_mask, 0] == 3.0
    assert compiled.terminal_mask[u_next]


class PriorityCRM(CountingRewardMachine):
    """CRM with overlapping transitions for priority testing."""

    @property
    def u_0(self) -> int:
        """Return the initial state."""
        return 0

    @property
    def c_0(self) -> tuple[int, ...]:
        """Return the initial counter configuration."""
        return (0,)

    def _get_state_transition_function(self) -> dict:
        """Return the state transition function."""
        return {
            0: {
                "EVENT_A / (-)": 1,
                "/ (-)": 2,
            }
        }

    def _get_counter_transition_function(self) -> dict:
        """Return the counter transition function."""
        return {
            0: {
                "EVENT_A / (-)": (0,),
                "/ (-)": (0,),
            }
        }

    def _get_reward_transition_function(self) -> dict:
        """Return the reward transition function."""
        return {
            0: {
                "EVENT_A / (-)": 1.0,
                "/ (-)": 2.0,
            }
        }

    def sample_counter_configurations(self) -> list[tuple[int, ...]]:
        """Return counter configurations."""
        return [(0,)]


class DynamicRewardCRM(CRM):
    """CRM with a dynamic Python reward callable."""

    def _get_reward_transition_function(self) -> dict:
        """Return the reward transition function."""
        rewards = super()._get_reward_transition_function()
        rewards[0]["EVENT_A / (-)"] = _dynamic_reward
        return rewards


class JaxDynamicRewardCRM(CRM):
    """CRM whose EVENT_A reward is a @jax_reward-marked callable."""

    def _get_reward_transition_function(self) -> dict:
        """Return the reward transition function."""
        rewards = super()._get_reward_transition_function()
        rewards[0]["EVENT_A / (-)"] = _jax_dynamic_reward
        return rewards


class SharedJaxRewardCRM(CRM):
    """CRM reusing one @jax_reward callable across two transitions."""

    def _get_reward_transition_function(self) -> dict:
        """Return the reward transition function."""
        rewards = super()._get_reward_transition_function()
        rewards[0]["EVENT_A / (-)"] = _jax_dynamic_reward
        rewards[1]["EVENT_A / (-)"] = _jax_dynamic_reward
        return rewards


class TerminalRewardMachine(RewardMachine):
    """Reward machine used to test implicit JAX CRM adaptation."""

    @property
    def u_0(self) -> int:
        """Return the initial state."""
        return 0

    def _get_state_transition_function(self) -> dict:
        """Return the state transition function."""
        return {
            0: {
                "EVENT_A": -1,
                "NOT EVENT_A": 0,
            }
        }

    def _get_reward_transition_function(self) -> dict:
        """Return the reward transition function."""
        return {
            0: {
                "EVENT_A": 3.0,
                "NOT EVENT_A": -0.1,
            }
        }


def _dynamic_reward(obs, action, next_obs) -> float:
    del action
    return float(next_obs[0] - obs[0])


@jax_reward
def _jax_dynamic_reward(obs, action, next_obs):
    """JAX-traceable reward: five times the change in the first obs element."""
    del action
    return (next_obs[0] - obs[0]) * 5.0


def _props_from_mask(env_props: tuple, prop_mask: int) -> set:
    return {prop for idx, prop in enumerate(env_props) if prop_mask & (1 << idx)}


def _counter_mask(c: tuple[int, ...]) -> int:
    return sum((1 if c_i != 0 else 0) << idx for idx, c_i in enumerate(c))
