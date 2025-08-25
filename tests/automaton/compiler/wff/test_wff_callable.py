import pytest

from pycrm.automaton.compiler import (
    _construct_callable_wff_expression_str_repr,
    _construct_wff_callable,
)
from tests.conftest import EnvProps


def wff_construction_cases() -> list[dict[str, str]]:
    """Returns test cases for WFF string construction.

    Returns:
        list[dict]: A list of test cases containing:
            - description (str): Description of the test case
            - wff (str): Input WFF expression
            - expected (str): Expected string representation
    """
    return [
        {
            "description": "Simple event check",
            "wff": "EVENT_A",
            "expected": "EnvProps.EVENT_A in props",
        },
        {
            "description": "Compound expression with AND and NOT",
            "wff": "EVENT_A and not EVENT_B",
            "expected": "EnvProps.EVENT_A in props and not EnvProps.EVENT_B in props",
        },
        {
            "description": "Nested expression with NOT",
            "wff": "not (EVENT_A and EVENT_B)",
            "expected": (
                "not (EnvProps.EVENT_A in props and EnvProps.EVENT_B in props)"
            ),
        },
        {
            "description": "Empty expression (tautology)",
            "wff": "",
            "expected": "True",
        },
        # Case insensitivity tests
        {
            "description": "Uppercase NOT operator",
            "wff": "EVENT_A NOT EVENT_B",
            "expected": "EnvProps.EVENT_A in props and not EnvProps.EVENT_B in props",
        },
        {
            "description": "Uppercase OR operator",
            "wff": "EVENT_A OR EVENT_B",
            "expected": "EnvProps.EVENT_A in props or EnvProps.EVENT_B in props",
        },
        {
            "description": "Uppercase AND operator",
            "wff": "EVENT_A AND EVENT_B",
            "expected": "EnvProps.EVENT_A in props and EnvProps.EVENT_B in props",
        },
        {
            "description": "Mixed case NOT operator",
            "wff": "EVENT_A Not EVENT_B",
            "expected": "EnvProps.EVENT_A in props and not EnvProps.EVENT_B in props",
        },
        {
            "description": "Mixed case OR operator",
            "wff": "EVENT_A Or EVENT_B",
            "expected": "EnvProps.EVENT_A in props or EnvProps.EVENT_B in props",
        },
        {
            "description": "Mixed case AND operator",
            "wff": "EVENT_A And EVENT_B",
            "expected": "EnvProps.EVENT_A in props and EnvProps.EVENT_B in props",
        },
        {
            "description": "Complex expression with uppercase operators",
            "wff": "NOT (EVENT_A OR EVENT_B)",
            "expected": "not (EnvProps.EVENT_A in props or EnvProps.EVENT_B in props)",
        },
        {
            "description": "Complex expression with mixed case operators",
            "wff": "Not (EVENT_A Or EVENT_B)",
            "expected": "not (EnvProps.EVENT_A in props or EnvProps.EVENT_B in props)",
        },
    ]


class TestWffCallableConstruction:
    """Tests for the WFF callable construction function."""

    @pytest.mark.parametrize(
        "test_case",
        wff_construction_cases(),
        ids=lambda test_case: test_case["description"],
    )
    def test_wff_expression_construction(self, test_case: dict[str, str]) -> None:
        """Tests construction of WFF expression string representations.

        Args:
            test_case (dict[str, str]): Test case containing WFF and expected result
        """
        assert (
            _construct_callable_wff_expression_str_repr(test_case["wff"], EnvProps)
            == test_case["expected"]
        )

    def test_wff_callable_construction_simple(self) -> None:
        """Tests construction and evaluation of a simple WFF callable."""
        wff_expr = "EVENT_A and not EVENT_B"
        wff_callable = _construct_wff_callable(wff_expr, EnvProps)
        assert wff_callable([]) is False
        assert wff_callable([EnvProps.EVENT_A]) is True
        assert wff_callable([EnvProps.EVENT_B]) is False
        assert wff_callable([EnvProps.EVENT_A, EnvProps.EVENT_B]) is False

    def test_wff_callable_construction_complex(self) -> None:
        """Tests construction and evaluation of a complex WFF callable."""
        wff_expr = "EVENT_A and not (EVENT_A and EVENT_B) or EVENT_B"
        wff_callable = _construct_wff_callable(wff_expr, EnvProps)
        assert wff_callable([]) is False
        assert wff_callable([EnvProps.EVENT_A]) is True
        assert wff_callable([EnvProps.EVENT_B]) is True
        assert wff_callable([EnvProps.EVENT_A, EnvProps.EVENT_B]) is True

    def test_wff_callable_construction_tautology(self) -> None:
        """Tests construction and evaluation of a tautological WFF callable."""
        wff_expr = ""
        wff_callable = _construct_wff_callable(wff_expr, EnvProps)
        assert wff_callable([]) is True
        assert wff_callable([EnvProps.EVENT_A]) is True
        assert wff_callable([EnvProps.EVENT_B]) is True
        assert wff_callable([EnvProps.EVENT_A, EnvProps.EVENT_B]) is True

    def test_wff_callable_construction_case_insensitive_not(self) -> None:
        """Test WFF callable construction with case-insensitive NOT."""
        wff_expr = "EVENT_A NOT EVENT_B"
        wff_callable = _construct_wff_callable(wff_expr, EnvProps)
        assert wff_callable([]) is False
        assert wff_callable([EnvProps.EVENT_A]) is True
        assert wff_callable([EnvProps.EVENT_B]) is False
        assert wff_callable([EnvProps.EVENT_A, EnvProps.EVENT_B]) is False

    def test_wff_callable_construction_case_insensitive_or(self) -> None:
        """Test construction and evaluation of WFF callable with case-insensitive OR."""
        wff_expr = "EVENT_A OR EVENT_B"
        wff_callable = _construct_wff_callable(wff_expr, EnvProps)
        assert wff_callable([]) is False
        assert wff_callable([EnvProps.EVENT_A]) is True
        assert wff_callable([EnvProps.EVENT_B]) is True
        assert wff_callable([EnvProps.EVENT_A, EnvProps.EVENT_B]) is True

    def test_wff_callable_construction_case_insensitive_and(self) -> None:
        """Test WFF callable construction with case-insensitive AND."""
        wff_expr = "EVENT_A AND EVENT_B"
        wff_callable = _construct_wff_callable(wff_expr, EnvProps)
        assert wff_callable([]) is False
        assert wff_callable([EnvProps.EVENT_A]) is False
        assert wff_callable([EnvProps.EVENT_B]) is False
        assert wff_callable([EnvProps.EVENT_A, EnvProps.EVENT_B]) is True

    def test_wff_callable_construction_mixed_case_operators(self) -> None:
        """Test WFF callable construction with mixed case operators."""
        wff_expr = "NOT (EVENT_A Or EVENT_B)"
        wff_callable = _construct_wff_callable(wff_expr, EnvProps)
        # This should be equivalent to: not (EVENT_A or EVENT_B)
        assert wff_callable([]) is True
        assert wff_callable([EnvProps.EVENT_A]) is False
        assert wff_callable([EnvProps.EVENT_B]) is False
        assert wff_callable([EnvProps.EVENT_A, EnvProps.EVENT_B]) is False
