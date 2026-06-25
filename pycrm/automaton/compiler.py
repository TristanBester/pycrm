from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, EnumMeta
from typing import (
    Callable,
    Collection,
    Literal,
    Mapping,
    Sequence,
    TypeAlias,
    cast,
)

import pyparsing as pp

CounterTest: TypeAlias = Literal["Z", "NZ", "-"]
EventExpression: TypeAlias = (
    "EventPredicate | Tautology | NotExpression | AndExpression | OrExpression"
)

_INVALID_TRANSITION_EXPRESSION = (
    "Invalid transition expression. Required format is "
    "'EVENT_FORMULA / COUNTER_STATES', "
    "e.g. 'EVENT_A and not EVENT_B / (Z,NZ)'."
)

pp.ParserElement.enable_packrat()


@dataclass(frozen=True)
class Tautology:
    """Event expression that always evaluates to true."""

    def _evaluate(
        self, props: Collection[Enum], event_members: Mapping[str, Enum]
    ) -> bool:
        del props, event_members
        return True

    def _event_names(self) -> set[str]:
        return set()


@dataclass(frozen=True)
class EventPredicate:
    """Event expression matching a single environment proposition."""

    name: str

    def _evaluate(
        self, props: Collection[Enum], event_members: Mapping[str, Enum]
    ) -> bool:
        return event_members[self.name] in props

    def _event_names(self) -> set[str]:
        return {self.name}


@dataclass(frozen=True)
class NotExpression:
    """Boolean negation expression."""

    operand: EventExpression

    def _evaluate(
        self, props: Collection[Enum], event_members: Mapping[str, Enum]
    ) -> bool:
        return not self.operand._evaluate(props, event_members)

    def _event_names(self) -> set[str]:
        return self.operand._event_names()


@dataclass(frozen=True)
class AndExpression:
    """Boolean conjunction expression."""

    left: EventExpression
    right: EventExpression

    def _evaluate(
        self, props: Collection[Enum], event_members: Mapping[str, Enum]
    ) -> bool:
        return self.left._evaluate(props, event_members) and self.right._evaluate(
            props, event_members
        )

    def _event_names(self) -> set[str]:
        return self.left._event_names() | self.right._event_names()


@dataclass(frozen=True)
class OrExpression:
    """Boolean disjunction expression."""

    left: EventExpression
    right: EventExpression

    def _evaluate(
        self, props: Collection[Enum], event_members: Mapping[str, Enum]
    ) -> bool:
        return self.left._evaluate(props, event_members) or self.right._evaluate(
            props, event_members
        )

    def _event_names(self) -> set[str]:
        return self.left._event_names() | self.right._event_names()


@dataclass(frozen=True)
class TransitionExpression:
    """Parsed CRM transition condition."""

    event_expression: EventExpression
    counter_tests: tuple[CounterTest, ...]


def compile_transition_expression(expression: str, env_props: EnumMeta) -> Callable:
    """Compile a transition expression into a callable.

    Args:
        expression (str): The transition expression to compile.
        env_props (EnumMeta): The environment property enum.

    Returns:
        A callable transition formula.
    """
    parsed_expression = _parse_transition_expression(expression, env_props)
    event_members = cast(Mapping[str, Enum], env_props.__members__)
    event_expression = parsed_expression.event_expression
    counter_tests = parsed_expression.counter_tests

    def transition_formula(
        props: Collection[Enum], counter_states: Sequence[int]
    ) -> bool:
        if len(counter_states) != len(counter_tests):
            raise ValueError(
                "Counter state arity mismatch. "
                + f"Expected {len(counter_tests)} values, "
                + f"received {len(counter_states)}."
            )

        return event_expression._evaluate(
            props, event_members
        ) and _counter_tests_match(counter_tests, counter_states)

    return transition_formula


def get_counter_test_arity(expression: str, env_props: EnumMeta) -> int:
    """Return the number of counter tests in a transition expression."""
    parsed_expression = _parse_transition_expression(expression, env_props)
    return len(parsed_expression.counter_tests)


def _parse_transition_expression(
    expression: str, env_props: EnumMeta
) -> TransitionExpression:
    if not expression.strip():
        raise ValueError(_INVALID_TRANSITION_EXPRESSION)

    grammar = _transition_expression_grammar()
    try:
        parsed = grammar.parse_string(expression, parse_all=True)
    except pp.ParseBaseException as exc:
        raise ValueError(_INVALID_TRANSITION_EXPRESSION) from exc

    event_expression = cast(
        EventExpression, parsed["event"] if "event" in parsed else Tautology()
    )
    counter_tests = cast(tuple[CounterTest, ...], tuple(parsed["counter_tests"]))
    _validate_event_names(event_expression, env_props)
    return TransitionExpression(event_expression, counter_tests)


def _transition_expression_grammar() -> pp.ParserElement:
    event_expression = _event_expression_grammar()
    counter_tests = _counter_tests_grammar()

    return (
        pp.Optional(event_expression, default=Tautology())("event")
        + pp.Suppress("/")
        + counter_tests("counter_tests")
        + pp.StringEnd()
    )


def _event_expression_grammar() -> pp.ParserElement:
    identifier = pp.Word(pp.alphas + "_", pp.alphanums + "_")
    identifier = identifier.set_parse_action(_event_predicate_action)
    tau = pp.CaselessKeyword("TAU").set_parse_action(lambda _: Tautology())

    atom = tau | identifier
    return pp.infix_notation(
        atom,
        [
            (
                pp.CaselessKeyword("not"),
                1,
                pp.opAssoc.RIGHT,
                lambda tokens: NotExpression(tokens[0][1]),
            ),
            (
                pp.CaselessKeyword("and"),
                2,
                pp.opAssoc.LEFT,
                lambda tokens: _fold_binary_expression(tokens[0], AndExpression),
            ),
            (
                pp.CaselessKeyword("or"),
                2,
                pp.opAssoc.LEFT,
                lambda tokens: _fold_binary_expression(tokens[0], OrExpression),
            ),
        ],
    )


def _counter_tests_grammar() -> pp.ParserElement:
    counter_test = pp.Keyword("NZ") | pp.Keyword("Z") | pp.Literal("-")
    return pp.Group(
        pp.Suppress("(") + pp.delimited_list(counter_test, min=1) + pp.Suppress(")")
    )


def _event_predicate_action(tokens: pp.ParseResults) -> EventPredicate:
    return EventPredicate(str(tokens[0]))


def _fold_binary_expression(tokens: pp.ParseResults, expression_type: type) -> object:
    expression = tokens[0]
    for index in range(2, len(tokens), 2):
        expression = expression_type(expression, tokens[index])
    return expression


def _validate_event_names(expression: EventExpression, env_props: EnumMeta) -> None:
    unknown_events = sorted(expression._event_names() - set(env_props.__members__))
    if unknown_events:
        event_list = ", ".join(unknown_events)
        raise ValueError(
            _INVALID_TRANSITION_EXPRESSION + f" Unknown event(s): {event_list}."
        )


def _counter_tests_match(
    counter_tests: tuple[CounterTest, ...], counter_states: Sequence[int]
) -> bool:
    for test, counter in zip(counter_tests, counter_states, strict=True):
        if test == "Z" and counter != 0:
            return False
        if test == "NZ" and counter == 0:
            return False

    return True
