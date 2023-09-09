###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
"""
Helper functions for the Maximum Cut (MaxCut) problem
"""
from qokit.fur.qaoa_simulator_base import TermsType
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


def get_maxcut_terms(G: nx.Graph) -> TermsType:
    """Get terms corresponding to cost function value

    .. math::

        S = \\sum_{(i,j,w)\\in G} w*(1-s_i*s_j)/2

    Args:
        G: MaxCut problem graph
    Returns:
        terms to be used in the simulation
    """
    if nx.is_weighted(G):
        terms = [(-float(G[u][v]["weight"]) / 2, (int(u), int(v))) for u, v, *_ in G.edges()]
        total_w = sum([float(G[u][v]["weight"]) for u, v, *_ in G.edges()])

    else:
        terms = [(-1 / 2, (int(e[0]), int(e[1]))) for e in G.edges()]
        total_w = int(G.number_of_edges())
    N = G.number_of_nodes()
    terms.append((+total_w / 2, tuple()))
    return terms


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
