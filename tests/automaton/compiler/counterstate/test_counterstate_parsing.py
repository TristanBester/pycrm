import pytest

from crm.automaton.compiler import _extract_counter_states


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
            "description": "Expression with zero counters",
            "expression": "EVENT_A and not EVENT_B / (Z, Z)",
            "expected": "(Z, Z)",
        },
        {
            "description": "Expression with non-zero counter and wildcard",
            "expression": "EVENT_A / (NZ, -)",
            "expected": "(NZ, -)",
        },
        {
            "description": "Complex expression with multiple counters",
            "expression": "not (EVENT_A and EVENT_B) / (-, -, NZ, Z)",
            "expected": "(-, -, NZ, Z)",
        },
        {
            "description": "Simple counter expression",
            "expression": "not (EVENT_A and EVENT_B) / X",
            "expected": "X",
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
        {
            "description": "Incomplete counter parentheses - left",
            "expression": "(Z, Z",
        },
        {
            "description": "Incomplete counter parentheses - right",
            "expression": "Z, Z)",
        },
        {
            "description": "Single counter value",
            "expression": "Z",
        },
    ]


class TestCounterStateParsing:
    """Test counter state parsing."""

    @pytest.mark.parametrize(
        "test_case",
        valid_expressions(),
        ids=lambda test_case: test_case["description"],
    )
    def test_counter_state_parsing_success(self, test_case: dict[str, str]) -> None:
        """Tests successful parsing of valid counter state expressions.

        Args:
            test_case (dict[str, str]): Test case containing:
                - expression: Input expression to parse
                - expected: Expected parsed result
        """
        assert _extract_counter_states(test_case["expression"]) == test_case["expected"]

    @pytest.mark.parametrize(
        "test_case",
        invalid_expressions(),
        ids=lambda test_case: test_case["description"],
    )
    def test_counter_state_parsing_failure(self, test_case: dict[str, str]) -> None:
        """Tests that invalid expressions raise appropriate errors.

        Args:
            test_case (dict[str, str]): Test case containing:
                - expression: Invalid input expression
        """
        with pytest.raises(ValueError) as exc_info:
            _extract_counter_states(test_case["expression"])
        assert "Invalid transition expression." in str(exc_info.value)

    def test_counter_state_parsing_stripped(self) -> None:
        """Tests that whitespace is properly stripped from the counter expression."""
        expression = "EVENT_A and not EVENT_B / (Z, Z) "
        assert _extract_counter_states(expression) == "(Z, Z)"
