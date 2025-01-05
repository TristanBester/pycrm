import networkx as nx


def compute_crm_transition_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    transitions = _compute_crm_transitions()

    for (_, u, _), v in transitions.items():
        graph.add_edge(u, v)
    return graph


def _compute_crm_transitions() -> dict:
    delta_u_0 = {
        (("M",), 0, ("-", "-")): 0,
        (("E",), 0, ("-", "-")): 1,
        (("C",), 0, ("-", "-")): 0,
        (("P",), 0, ("-", "-")): 0,
        (("*",), 0, ("-", "-")): 3,
        ((), 0, ("-", "-")): 0,
    }
    delta_u_1 = {
        (("M",), 1, ("-", "-")): 1,
        (("E",), 1, ("-", "-")): 3,
        (("C",), 1, ("-", "-")): 1,
        (("P",), 1, ("-", "-")): 1,
        (("*",), 1, ("-", "-")): 3,
        ((), 1, ("NZ", "-")): 1,
        ((), 1, ("Z", "-")): 2,
    }
    delta_u_2 = {
        (("M",), 2, ("-", "-")): 2,
        (("E",), 2, ("-", "-")): 2,
        (("C",), 2, ("-", "-")): 2,
        (("P",), 2, ("-", "-")): 2,
        (("*",), 2, ("-", "-")): 3,
        ((), 2, ("-", "NZ")): 2,
        ((), 2, ("-", "Z")): 3,
    }
    delta_u = delta_u_0 | delta_u_1 | delta_u_2
    return delta_u
