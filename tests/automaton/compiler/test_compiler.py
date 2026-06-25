import inspect

import pytest

from pycrm.automaton import RewardMachine, RmToCrmAdapter
from pycrm.automaton.compiler import compile_transition_expression
from tests.conftest import EnvProps


class AdapterRewardMachine(RewardMachine):
    """Reward machine used to verify RM-to-CRM parser integration."""

    def __init__(self) -> None:
        """Initialise the reward machine."""
        super().__init__(env_prop_enum=EnvProps)

    @property
    def u_0(self) -> int:
        """Return the initial state."""
        return 0

    def _get_state_transition_function(self) -> dict:
        """Return the state transition function."""
        return {
            0: {
                "EVENT_A": 1,
                "not EVENT_A": 0,
            }
        }

    def _get_reward_transition_function(self) -> dict:
        """Return the reward transition function."""
        return {
            0: {
                "EVENT_A": 1,
                "not EVENT_A": 0,
            }
        }


@pytest.mark.parametrize(
    ("expression", "props", "counter_states", "expected"),
    [
        ("EVENT_A / (Z)", [EnvProps.EVENT_A], [0], True),
        ("EVENT_A / (Z)", [EnvProps.EVENT_A], [1], False),
        ("EVENT_A / (NZ)", [EnvProps.EVENT_A], [3], True),
        ("EVENT_A / (NZ)", [EnvProps.EVENT_A], [0], False),
        ("EVENT_A / (NZ)", [EnvProps.EVENT_A], [-1], True),
        ("EVENT_A / (-)", [EnvProps.EVENT_A], [0], True),
        ("EVENT_A / (-)", [EnvProps.EVENT_A], [10], True),
        ("EVENT_A and not EVENT_B / (Z,NZ)", [EnvProps.EVENT_A], [0, 1], True),
        (
            "EVENT_A and not EVENT_B / (Z,NZ)",
            [EnvProps.EVENT_A, EnvProps.EVENT_B],
            [0, 1],
            False,
        ),
        ("not (EVENT_A or EVENT_B) / (Z)", [], [0], True),
        ("not (EVENT_A or EVENT_B) / (Z)", [EnvProps.EVENT_A], [0], False),
        ("(EVENT_A and EVENT_B) or EVENT_B / (Z)", [EnvProps.EVENT_B], [0], True),
        ("/ (Z,-)", [EnvProps.EVENT_A], [0, 1], True),
        ("/ (Z,-)", [EnvProps.EVENT_A], [1, 1], False),
        ("TAU / (-)", [], [1], True),
        ("tau / (-)", [], [1], True),
        ("EVENT_A Or EVENT_B / (Z)", [EnvProps.EVENT_B], [0], True),
        ("EVENT_A AND NOT EVENT_B / (Z)", [EnvProps.EVENT_A], [0], True),
    ],
)
def test_compile_transition_expression(
    expression: str,
    props: list[EnvProps],
    counter_states: list[int],
    expected: bool,
) -> None:
    """Test transition expressions accepted by the CRM grammar."""
    transition_callable = compile_transition_expression(expression, EnvProps)
    assert transition_callable(props, counter_states) is expected


@pytest.mark.parametrize(
    "expression",
    [
        "",
        "EVENT_A",
        "EVENT_A /",
        "EVENT_A / Z",
        "EVENT_A / ()",
        "EVENT_A / (z)",
        "UNKNOWN / (Z)",
        "len(props) == 0 / (Z)",
        "__import__('os') / (Z)",
        "EVENT_A / (Z) extra",
        "EVENT_A / (Z) / (Z)",
        "EVENT_A NOT EVENT_B / (Z)",
        "42 / (Z)",
    ],
)
def test_compile_transition_expression_rejects_invalid_dsl(
    expression: str,
) -> None:
    """Test expressions outside the CRM grammar are rejected."""
    with pytest.raises(ValueError) as exc_info:
        compile_transition_expression(expression, EnvProps)

    assert "Invalid transition expression" in str(exc_info.value)


def test_compile_transition_expression_rejects_counter_arity_mismatch() -> None:
    """Test counter vectors must match the parsed zero-test vector."""
    transition_callable = compile_transition_expression("EVENT_A / (Z,NZ)", EnvProps)

    with pytest.raises(ValueError) as exc_info:
        transition_callable([EnvProps.EVENT_A], [0])

    assert "Counter state arity mismatch" in str(exc_info.value)


def test_compile_transition_expression_returns_expected_signature() -> None:
    """Test the compiled callable has the expected public parameter names."""
    transition_callable = compile_transition_expression("EVENT_A / (Z)", EnvProps)

    assert callable(transition_callable)
    assert list(inspect.signature(transition_callable).parameters) == [
        "props",
        "counter_states",
    ]


def test_reward_machine_adapter_converts_rm_expressions_to_crm_dsl() -> None:
    """Test RM event formulas still work through the adapter."""
    adapter = RmToCrmAdapter(AdapterRewardMachine())

    u_next, c_next, reward_fn = adapter.transition(0, (0,), {EnvProps.EVENT_A})

    assert u_next == 1
    assert c_next == (0,)
    assert reward_fn(None, None, None) == 1.0
