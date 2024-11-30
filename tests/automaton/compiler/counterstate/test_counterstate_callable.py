from typing import Any

import pytest

from crm.automaton.compiler import (
    _construct_callable_counter_state_str_repr,
    _construct_counter_state_callable,
)


def valid_counter_states() -> list[dict[str, Any]]:
    """Returns a list of valid counter state expressions for testing.

    Returns:
        list[dict]: A list of test cases, where each case contains:
            - description (str): Description of the test case
            - counter_states (str): Input counter state expression
            - expected_result (str): Expected string representation
            - test_values (list[list[int]]): List of counter values to test
            - expected_outcomes (list[bool]): Expected outcomes for test values
    """
    return [
        {
            "description": "Single counter zero state",
            "counter_states": "(Z)",
            "expected_result": "counters[0] == 0",
            "test_values": [[0], [1]],
            "expected_outcomes": [True, False],
        },
        {
            "description": "Single counter non-zero state",
            "counter_states": "(NZ)",
            "expected_result": "counters[0] == 1",
            "test_values": [[0], [1]],
            "expected_outcomes": [False, True],
        },
        {
            "description": "Single counter any state",
            "counter_states": "(-)",
            "expected_result": "True",
            "test_values": [[0], [1]],
            "expected_outcomes": [True, True],
        },
        {
            "description": "Multiple counters all zero",
            "counter_states": "(Z, Z)",
            "expected_result": "counters[0] == 0 and counters[1] == 0",
            "test_values": [[0, 0], [1, 0], [0, 1], [1, 1]],
            "expected_outcomes": [True, False, False, False],
        },
        {
            "description": "Multiple counters zero and non-zero",
            "counter_states": "(Z, NZ)",
            "expected_result": "counters[0] == 0 and counters[1] == 1",
            "test_values": [[0, 0], [1, 0], [0, 1], [1, 1]],
            "expected_outcomes": [False, False, True, False],
        },
        {
            "description": "Multiple counters non-zero and zero",
            "counter_states": "(NZ, Z)",
            "expected_result": "counters[0] == 1 and counters[1] == 0",
            "test_values": [[0, 0], [1, 0], [0, 1], [1, 1]],
            "expected_outcomes": [False, True, False, False],
        },
        {
            "description": "Multiple counters all non-zero",
            "counter_states": "(NZ, NZ)",
            "expected_result": "counters[0] == 1 and counters[1] == 1",
            "test_values": [[0, 0], [1, 0], [0, 1], [1, 1]],
            "expected_outcomes": [False, False, False, True],
        },
        {
            "description": "Multiple counters any and non-zero",
            "counter_states": "(-, NZ)",
            "expected_result": "True and counters[1] == 1",
            "test_values": [[0, 0], [1, 0], [0, 1], [1, 1]],
            "expected_outcomes": [False, False, True, True],
        },
        {
            "description": "Multiple counters zero and any",
            "counter_states": "(Z, -)",
            "expected_result": "counters[0] == 0 and True",
            "test_values": [[0, 0], [1, 0], [0, 1], [1, 1]],
            "expected_outcomes": [True, False, True, False],
        },
        {
            "description": "Multiple counters all any",
            "counter_states": "(-, -)",
            "expected_result": "True and True",
            "test_values": [[0, 0], [1, 0], [0, 1], [1, 1]],
            "expected_outcomes": [True, True, True, True],
        },
    ]


class TestCounterStateCallableConstruction:
    """Tests for counter state callable construction and evaluation."""

    @pytest.mark.parametrize(
        "test_case",
        valid_counter_states(),
        ids=lambda test_case: test_case["description"],
    )
    def test_counter_state_expression_construction(self, test_case: dict) -> None:
        """Tests construction of counter state string representations.

        Args:
            test_case (dict): Test case containing counter state expression
                and expected result
        """
        assert (
            _construct_callable_counter_state_str_repr(test_case["counter_states"])
            == test_case["expected_result"]
        )

    @pytest.mark.parametrize(
        "test_case",
        valid_counter_states(),
        ids=lambda test_case: test_case["description"],
    )
    def test_counter_state_callable_evaluation(self, test_case: dict) -> None:
        """Tests evaluation of counter state callables with various inputs.

        Args:
            test_case (dict): Test case containing counter states and test values
        """
        counter_state_callable = _construct_counter_state_callable(
            test_case["counter_states"]
        )

        for test_value, expected_outcome in zip(
            test_case["test_values"],
            test_case["expected_outcomes"],
            strict=True,
        ):
            assert counter_state_callable(test_value) is expected_outcome

    def test_counter_state_callable_invalid_counter_states(self) -> None:
        """Test that invalid counter states raise an error."""
        with pytest.raises(ValueError) as exc_info:
            _construct_counter_state_callable("(X, Z)")

        assert "Invalid counter expression" in str(exc_info.value)
