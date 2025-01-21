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
        G.edges[edge[0], edge[1]]['weight'] = 1 #J[edge[0], edge[1]]

    def sk_obj_simple(x, G):
        cut = 0
        for i, j in G.edges():
            if x[i] != x[j]:
                # the edge is cut
                cut += 1
        return cut

    x = np.random.choice([0, 1], G.number_of_nodes())
    assert np.isclose(sk_obj(x, w=get_adjacency_matrix(G)), sk_obj_simple(x, G))
