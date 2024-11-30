import pytest

from crm.automaton.compiler import compile_transition_expression
from tests.conftest import EnvProps


class TestTransitionExpressionCompilation:
    """Tests for the transition expression compilation function."""

    def test_simple_transition_expression(self) -> None:
        """Test compilation of transition expression with single event and counter.

        Tests a basic transition expression "EVENT_A / (Z)" which should evaluate
        to True only when EVENT_A is present and the counter value is 0.
        """
        transition_expr = "EVENT_A / (Z)"
        transition_callable = compile_transition_expression(transition_expr, EnvProps)
        assert transition_callable([EnvProps.EVENT_A], [0]) is True
        assert transition_callable([EnvProps.EVENT_A], [1]) is False
        assert transition_callable([EnvProps.EVENT_B], [0]) is False
        assert transition_callable([EnvProps.EVENT_B], [1]) is False

    def test_complex_transition_expression(self) -> None:
        """Test compilation of a complex transition expression with multiple conditions.

        Tests the transition expression "EVENT_A and not EVENT_B / (Z, NZ)" which should
        evaluate to True only when EVENT_A is present without EVENT_B, and the first
        counter is 0 while the second is non-zero.
        """
        transition_expr = "EVENT_A and not EVENT_B / (Z, NZ)"
        transition_callable = compile_transition_expression(transition_expr, EnvProps)
        assert (
            transition_callable([EnvProps.EVENT_A, EnvProps.EVENT_B], [0, 1]) is False
        )
        assert (
            transition_callable([EnvProps.EVENT_A, EnvProps.EVENT_B], [1, 0]) is False
        )
        assert transition_callable([EnvProps.EVENT_A], [0, 0]) is False
        assert transition_callable([EnvProps.EVENT_A], [0, 1]) is True
        assert transition_callable([EnvProps.EVENT_B], [0, 0]) is False
        assert transition_callable([EnvProps.EVENT_B], [0, 1]) is False

    def test_tautological_transition_expression(self) -> None:
        """Test compilation of a tautological transition expression.

        Tests the transition expression "/ (Z, -)" which should evaluate to True
        regardless of events present, as long as the first counter is 0 (second counter
        is ignored due to '-' wildcard).
        """
        transition_expr = "/ (Z, -)"
        transition_callable = compile_transition_expression(transition_expr, EnvProps)
        assert transition_callable([], [0, 0]) is True
        assert transition_callable([], [0, 1]) is True
        assert transition_callable([], [1, 1]) is False
        assert transition_callable([EnvProps.EVENT_A], [0, 1]) is True
        assert transition_callable([EnvProps.EVENT_B], [0, 1]) is True
        assert transition_callable([EnvProps.EVENT_A, EnvProps.EVENT_B], [0, 1]) is True

    def test_invalid_wff_transition_expression(self) -> None:
        """Test compilation fails for invalid well-formed formula transition expression.

        Tests that compiling an expression without counter conditions ("EVENT_A and
        not EVENT_B") raises a ValueError with appropriate message.

        Raises:
            ValueError: When transition expression is missing counter conditions.
        """
        transition_expr = "EVENT_A and not EVENT_B"
        with pytest.raises(ValueError) as exc_info:
            compile_transition_expression(transition_expr, EnvProps)
        assert "Invalid transition expression." in str(exc_info.value)

    def test_invalid_counter_state_transition_expression(self) -> None:
        """Test compilation fails for invalid counter state specification.

        Tests that compiling an expression with improperly formatted counter conditions
        ("EVENT_A / Z" instead of "EVENT_A / (Z)") raises a ValueError.

        Raises:
            ValueError: When counter state specification is improperly formatted.
        """
        transition_expr = "EVENT_A / Z"
        with pytest.raises(ValueError) as exc_info:
            compile_transition_expression(transition_expr, EnvProps)
        assert "Invalid transition expression." in str(exc_info.value)
