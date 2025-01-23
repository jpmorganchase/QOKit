###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import numpy as np
import networkx as nx

from qokit.sk import sk_obj, get_adjacency_matrix, get_sk_terms


def test_sk_obj(n=5):
    G = nx.complete_graph(n)
    J = np.random.randn(n, n)
    J = (J + J.T)/2

    for edge in G.edges:
        G.edges[edge[0], edge[1]]['weight'] = J[edge[0], edge[1]]

    def sk_obj_simple(x, G):
        obj = 0
        w=get_adjacency_matrix(G)
        for i, j in G.edges():
                obj += w[i, j] * (2*x[i] - 1) * (2*x[j] - 1)
        return -2*obj/np.sqrt(n)

    x = np.random.choice([0, 1], G.number_of_nodes())
    assert np.isclose(sk_obj(x, w=get_adjacency_matrix(G)), sk_obj_simple(x, G))
