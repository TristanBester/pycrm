import re
import textwrap
from enum import EnumMeta
from typing import Callable


def compile_transition_expression(expression: str, env_props: EnumMeta) -> Callable:
    """Compile a transition expression into a callable.

    Args:
        expression (str): The transition expression to compile.
        env_props (EnumMeta): The environment property enum.

    Returns:
        A callable transition formula.
    """
    wff_expr = _extract_wff(expression)
    counter_state_expr = _extract_counter_states(expression)

    wff_callable = _construct_wff_callable(wff_expr, env_props)
    counter_state_callable = _construct_counter_state_callable(counter_state_expr)

    func_template = textwrap.dedent("""
    def transition_formula(props: list[EnumMeta], counter_states: list[int]) -> bool:
        return wff_callable(props) and counter_state_callable(counter_states)
    """)
    local_namespace = {}
    global_namespace = {
        "wff_callable": wff_callable,
        "counter_state_callable": counter_state_callable,
        "EnumMeta": EnumMeta,
    }
    exec(func_template, global_namespace, local_namespace)
    transition_formula = local_namespace["transition_formula"]
    return transition_formula


def _extract_wff(expression: str) -> str:
    if "/" not in expression:
        raise ValueError(
            "Invalid transition expression. "
            "Required format is 'WFF / COUNTER_STATS', "
            "e.g. 'EVENT_A and not EVENT_B / (Z,NZ)'"
        )
    return expression.split("/")[0].strip()


def _extract_counter_states(expression: str) -> str:
    if "/" not in expression or not ("(" in expression and ")" in expression):
        raise ValueError(
            "Invalid transition expression. "
            "Required format is 'WFF / COUNTER_STATS', "
            "e.g. 'EVENT_A and not EVENT_B / (Z,NZ)'"
        )
    return expression.split("/")[1].strip()


def _construct_callable_wff_expression_str_repr(
    wff_expr: str, env_props: EnumMeta
) -> str:
    enum_name = env_props.__name__

    if wff_expr == "":
        return "True"

    # Preprend enum name to each proposition
    wff_expr = re.sub(r"(([A-Z]+_*)+)", rf"{enum_name}.\1", wff_expr)
    # Add "in props" after each "Proposition
    wff_expr = re.sub(rf"{enum_name}\.(\w+)", rf"{enum_name}.\1 in props", wff_expr)
    # Replace negation syntax
    wff_expr = re.sub(rf"not {enum_name}\.(\w+)", rf"{enum_name}.\1 not", wff_expr)
    return wff_expr


def _construct_wff_callable(wff_expr: str, env_props: EnumMeta) -> Callable:
    wff_expr = _construct_callable_wff_expression_str_repr(wff_expr, env_props)
    func_template = textwrap.dedent(f"""
    def wff(props):
        return {wff_expr}
    """)

    local_namespace = {}
    global_namespace = {env_props.__name__: env_props}
    exec(func_template, global_namespace, local_namespace)
    wff = local_namespace["wff"]
    return wff


def _construct_callable_counter_state_str_repr(counter_states: str) -> str:
    counter_expr = counter_states.replace(" ", "")
    counter_expr = counter_expr.replace("(", "").replace(")", "")
    condition_ls = counter_expr.split(",")

    conditions = []
    for i, c in enumerate(condition_ls):
        if c == "Z":
            conditions.append(f"counters[{i}] == 0")
        elif c == "NZ":
            conditions.append(f"counters[{i}] == 1")
        elif c == "-":
            conditions.append("True")
        else:
            raise ValueError(f"Invalid counter expression {c}.")

    conditions = " and ".join(conditions)
    return conditions


def _construct_counter_state_callable(counter_states: str) -> Callable:
    counter_expr = _construct_callable_counter_state_str_repr(counter_states)
    func_template = textwrap.dedent(f"""
    def counter_conditions(counters):
        counters = [0 if c == 0 else 1 for c in counters]
        return {counter_expr}
    """)

    local_namespace = {}
    global_namespace = {}
    exec(func_template, global_namespace, local_namespace)
    counter_conditions = local_namespace["counter_conditions"]
    return counter_conditions
