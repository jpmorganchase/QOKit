###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import numpy as np
import networkx as nx
from qokit.generator import (
    get_energy_term_indices_SK,
    get_graph_SK,
    get_energy_term_indices_max_q_xor,
)


def test_get_energy_term_indices_max_q_xor():
    # Test for spesific problem size and seed

    N = 5
    seed = 1
    q = 4
    d = 3
    terms = get_energy_term_indices_max_q_xor(N, q, d, seed)

    # Check the length of returned term

    assert len(terms) == 14
    # Check the format of each term in the list
    for term in terms:
        assert isinstance(term, tuple)
        assert len(terms) == 14
        assert isinstance(terms[1], tuple)
        assert all(isinstance(index, int) for index in term[1])

    # Test the different problem size,q, d and seed

    N = 10
    seed = 10
    q = 8
    d = 6
    terms = get_energy_term_indices_max_q_xor(N, q, d, seed)
    assert len(terms) == 67


def test_get_graph_SK():
    N = 5
    seed = 1
    G = get_graph_SK(N, seed)

    # Check the graph type
    assert isinstance(G, nx.Graph)

    # Check the number of nodes and edges in graph
    assert G.number_of_nodes() == 5
    assert G.number_of_edges() == 10

    # Check the weight attribute of each edge
    for u, v, data in G.edges(data=True):
        assert "weight" in data
        assert isinstance(data["weight"], float)

    # Test the different problem size and seed
    N = 10
    seed = 10
    G = get_graph_SK(N, seed)

    # Check the number of nodes and edges in graph
    assert G.number_of_nodes() == 10
    assert G.number_of_edges() == 45


def test_get_energy_term_indices_SK():
    N = 5
    seed = 1
    terms = get_energy_term_indices_SK(N, seed)
    assert len(terms) == 10

    # Check the format of each term in the list
    for term in terms:
        assert isinstance(term, tuple)
        assert isinstance(terms[1], tuple)
        assert all(isinstance(index, int) for index in term[1])

    # Test the different problem size and seed

    N = 10
    seed = 10
    terms = get_energy_term_indices_SK(N, seed)
    assert len(terms) == 45


def test_generators():
    """
    Only checks that the generators return non-gibberish and that the seed is respected
    """

    N = 5
    seed = 1
    terms1 = get_energy_term_indices_SK(N, seed=seed)
    _ = get_energy_term_indices_SK(N, seed=seed + 1)
    terms2 = get_energy_term_indices_SK(N, seed=seed)
    assert np.allclose(np.array([x[0] for x in terms1]), np.array([x[0] for x in terms2]))

    G = get_graph_SK(N, seed)
    assert np.allclose(
        np.array([x[0] for x in terms1]),
        np.array([G[u][v]["weight"] for _, (u, v) in terms1]),
    )

    N = 10
    q = 4
    d = 3
    seed = 1
    terms1 = get_energy_term_indices_max_q_xor(N, q, d, seed=seed)
    _ = get_energy_term_indices_max_q_xor(N, q, d, seed=seed + 1)
    terms2 = get_energy_term_indices_max_q_xor(N, q, d, seed=seed)
    for (J1, term1), (J2, term2) in zip(terms1, terms2):
        assert np.isclose(J1, J2)
        assert set(term1) == set(term2)
