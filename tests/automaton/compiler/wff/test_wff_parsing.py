import pytest

from crm.automaton.compiler import _extract_wff


def valid_expressions() -> list[dict[str, str]]:
    """Returns a list of valid expressions for testing.

    Returns:
        list[dict]: A list of test cases, where each case contains:
            - description (str): Description of the test case
            - expression (str): Input expression to test
            - expected (str): Expected output after parsing
    """
    return [
        {
            "description": "Complex expression with zero counters",
            "expression": "EVENT_A and not EVENT_B / (Z, Z)",
            "expected": "EVENT_A and not EVENT_B",
        },
        {
            "description": "Simple expression with non-zero counter and wildcard",
            "expression": "EVENT_A / (NZ,-)",
            "expected": "EVENT_A",
        },
        {
            "description": "Nested expression with negation and wildcard counters",
            "expression": "not (EVENT_A and EVENT_B) / (-,-)",
            "expected": "not (EVENT_A and EVENT_B)",
        },
        {
            "description": "Expression without counters",
            "expression": "X / ",
            "expected": "X",
        },
        {
            "description": "Tautological expression with zero counter",
            "expression": "/(Z)",
            "expected": "",
        },
    ]


def invalid_expressions() -> list[dict[str, str]]:
    """Returns a list of invalid expressions for testing.

    Returns:
        list[dict]: A list of test cases, where each case contains:
            - description (str): Description of the test case
            - expression (str): Invalid input expression to test
    """
    return [
        {
            "description": "Missing counter expression",
            "expression": "EVENT_A and not EVENT_B",
        },
        {
            "description": "Invalid format",
            "expression": "EVENT_A and not EVENT_B (Z, Z, Z)",
        },
        {
            "description": "Empty expression",
            "expression": "",
        },
        {
            "description": "Counters only",
            "expression": "(Z, Z)",
        },
    ]


class TestWffParsing:
    """Tests for the WFF parsing function."""

    @pytest.mark.parametrize(
        "test_case",
        valid_expressions(),
        ids=lambda test_case: test_case["description"],
    )
    def test_wff_expression_parsing_success(self, test_case: dict[str, str]) -> None:
        """Tests successful parsing of valid well-formed formula expressions.

        Args:
            test_case (dict[str, str]): Test case containing:
                - expression: Input expression to parse
                - expected_result: Expected parsed result
        """
        assert _extract_wff(test_case["expression"]) == test_case["expected"]

    @pytest.mark.parametrize(
        "test_case",
        invalid_expressions(),
        ids=lambda test_case: test_case["description"],
    )
    def test_wff_expression_parsing_failure(self, test_case: dict[str, str]) -> None:
        """Tests that invalid expressions raise appropriate errors.

        Args:
            test_case (dict[str, str]): Test case containing:
                - expression: Invalid input expression
        """
        with pytest.raises(ValueError) as exc_info:
            _extract_wff(test_case["expression"])
        assert "Invalid transition expression." in str(exc_info.value)

    def test_wff_expression_stripped(self) -> None:
        """Tests that counter expressions are properly stripped from the input."""
        expression = "EVENT_A and not EVENT_B / (Z, Z)"
        assert _extract_wff(expression) == "EVENT_A and not EVENT_B"
