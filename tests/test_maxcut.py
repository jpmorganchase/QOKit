###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import numpy as np
import networkx as nx

from qokit.maxcut import maxcut_obj, get_adjacency_matrix, get_maxcut_terms


def test_maxcut_obj():
    G = nx.Graph()
    G.add_edges_from([[0, 3], [0, 4], [1, 3], [1, 4], [2, 3], [2, 4]])

    def maxcut_obj_simple(x, G):
        cut = 0
        for i, j in G.edges():
            if x[i] != x[j]:
                # the edge is cut
                cut += 1
        return cut

    x = np.random.choice([0, 1], G.number_of_nodes())
    assert np.isclose(maxcut_obj(x, w=get_adjacency_matrix(G)), maxcut_obj_simple(x, G))
