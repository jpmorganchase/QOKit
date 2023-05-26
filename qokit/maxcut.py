"""
Helper functions for the Maximum Cut (MaxCut) problem
"""
import numpy as np
import networkx as nx


def maxcut_obj(x: np.ndarray, w: np.ndarray) -> float:
    """Compute the value of a cut.
    Args:
        x (numpy.ndarray): binary string as numpy array.
        w (numpy.ndarray): adjacency matrix returned by get_adjacency_matrix
    Returns:
        float: value of the cut.
    """
    X = np.outer(x, (1 - x))
    return np.sum(w * X)  # type: ignore


def get_adjacency_matrix(G: nx.Graph) -> np.ndarray:
    """Get adjacency matrix to be used in maxcut_obj
    Args:
        G (nx.Graph) : graph
    Returns:
        w (numpy.ndarray): adjacency matrix
    """
    n = G.number_of_nodes()
    w = np.zeros([n, n])

    for e in G.edges():
        if nx.is_weighted(G):
            w[e[0], e[1]] = G[e[0]][e[1]]["weight"]
            w[e[1], e[0]] = G[e[0]][e[1]]["weight"]
        else:
            w[e[0], e[1]] = 1
            w[e[1], e[0]] = 1
    return w
