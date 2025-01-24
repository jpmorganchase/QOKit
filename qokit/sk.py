###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
"""
Helper functions for the S_k problem
"""
from qokit.fur.qaoa_simulator_base import TermsType
import numpy as np
import networkx as nx


def sk_obj(x: np.ndarray, J: np.ndarray) -> float:
    """Compute the value of a cut.
    Args:
        x (numpy.ndarray): binary string as numpy array.
        w (numpy.ndarray): adjacency matrix returned by get_adjacency_matrix
    Returns:
        float: value of the cut.
    """
    n = len(x)
    X = np.outer(2 * x - 1, 2 * x - 1)
    return -np.sum(J * X) / np.sqrt(n)  # type: ignore


def get_sk_terms(J: np.ndarray) -> TermsType:
    """Get terms corresponding to cost function value

    .. math::

        S = \\sum_{(i,j)\\in G} J_ij * (s_i*s_j)/2

    Args:
        G: MaxCut problem graph
    Returns:
        terms to be used in the simulation
    """
    N = J.shape[0]
    G = nx.complete_graph(N)
    for edge in G.edges:
        G.edges[edge[0], edge[1]]["weight"] = J[edge[0], edge[1]]

    terms = [(-2 * float(G[u][v]["weight"]) / np.sqrt(N), (int(u), int(v))) for u, v, *_ in G.edges()]
    return terms


def get_adjacency_matrix(G: nx.Graph) -> np.ndarray:
    """Get adjacency matrix to be used in sk_obj
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
