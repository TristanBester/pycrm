import networkx as nx
import scipy


def cyclomatic_complexity(graph: nx.DiGraph) -> int:
    """Compute the cyclomatic complexity of a graph."""
    # Number of edges
    n_edges = graph.number_of_edges()
    # Number of nodes
    n_nodes = graph.number_of_nodes()
    # Number of connected components
    adj = nx.adjacency_matrix(graph).todense()
    n_components, _ = scipy.sparse.csgraph.connected_components(adj, directed=True)
    # Cyclomatic complexity formula
    cyclomatic_complexity = n_edges - n_nodes + 2 * n_components
    return cyclomatic_complexity
