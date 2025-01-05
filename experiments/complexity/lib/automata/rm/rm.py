import networkx as nx


def compute_rm_transition_graph(max_n: int = 3) -> nx.DiGraph:
    """Compute the RM transition graph."""
    graph = nx.DiGraph()
    transitions = _compute_rm_transitions(max_n)

    for (_, u), v in transitions.items():
        graph.add_edge(u, v)
    return graph


def _compute_rm_transitions(max_n: int = 3) -> dict:
    """Compute the RM transitions."""
    delta_u = {
        (("M",), 0): 1,
        (("E",), 0): 0,
        (("C",), 0): 0,
        (("P",), 0): 0,
        (("*",), 0): -1,
        ((), 0): 0,
    }
    state_counter = 1

    for i in range(1, max_n + 1):
        delta_u |= {
            (("M",), state_counter): state_counter + 1 + 2 * i,
            (("E",), state_counter): state_counter + 1,
            (("C",), state_counter): state_counter,
            (("P",), state_counter): state_counter,
            (("*",), state_counter): -1,
            ((), state_counter): state_counter,
        }
        state_counter += 1

        for _ in range(i):
            delta_u |= {
                (("M",), state_counter): state_counter,
                (("E",), state_counter): state_counter,
                (("C",), state_counter): state_counter + 1,
                (("P",), state_counter): state_counter,
                (("*",), state_counter): -1,
                ((), state_counter): state_counter,
            }
            state_counter += 1

        for _ in range(i):
            delta_u |= {
                (("M",), state_counter): state_counter,
                (("E",), state_counter): state_counter,
                (("C",), state_counter): state_counter,
                (("P",), state_counter): state_counter + 1,
                (("*",), state_counter): -1,
                ((), state_counter): state_counter,
            }
            state_counter += 1

        delta_u[(("P",), state_counter - 1)] = -1

    for i in delta_u:
        if delta_u[i] == -1:
            delta_u[i] = state_counter
    return delta_u
