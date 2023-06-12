###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import numpy as np
import networkx as nx
import random
from itertools import combinations
from typing import Optional


def get_energy_term_indices_SK(N: int, seed: Optional[int] = None):
    """Return indices of Pauli Zs and coefficients in front of them
    for the SK model
    Parameters
    ----------
    N : int
        Problem size (number of spins)
    seed : int
        Random seed
    Returns
    -------
    terms : list of tuples
        List of tuples, where each tuple defines a coefficient
        and a summand and contains indices of the Pauli Zs in the product
        e.g. if terms = [(0.5, (0,1)), (0.7, (0,1,2,3)), (-2, (1,2))]
        the Hamiltonian is 0.5*Z0Z1 + 0.7*Z0Z1Z2Z3 - 2*Z1Z2
    """
    rng = np.random.RandomState(seed)
    indices = list(combinations(range(N), 2))
    all_terms = []
    for term in indices:
        all_terms.append((rng.normal(0, 1) / np.sqrt(N), term))
    return all_terms


def get_graph_SK(N: int, seed: Optional[int] = None):
    """Convenience function for interfacing with QAOAKit
    Parameters
    ----------
    N : int
        Problem size (number of spins)
    seed : int
        Random seed
    Returns
    -------
    terms : list of tuples
        List of tuples, where each tuple defines a coefficient
        and a summand and contains indices of the Pauli Zs in the product
        e.g. if terms = [(0.5, (0,1)), (0.7, (0,1,2,3)), (-2, (1,2))]
        the Hamiltonian is 0.5*Z0Z1 + 0.7*Z0Z1Z2Z3 - 2*Z1Z2
    """
    all_terms = get_energy_term_indices_SK(N, seed=seed)
    G = nx.Graph()
    for J, (u, v) in all_terms:
        G.add_edge(u, v, weight=J)
    return G


def get_energy_term_indices_max_q_xor(N: int, q: int, d: int, seed: Optional[int] = None):
    """Return indices of Pauli Zs and coefficients in front of them
    for the Max-q-XORSAT on a random Erdos-Renyi directed multi-hypergraph problem
    Follows the definition at the bottom of page 4 of http://arxiv.org/abs/2204.10306v2
    Parameters
    ----------
    N : int
        Problem size (number of spins)
    q : int
        Size of each hyperedge (number of spins in each term)
    d : int
        Degree of the hypergraph
    seed : int
        Random seed
    Returns
    -------
    terms : list of tuples
        List of tuples, where each tuple defines a coefficient
        and a summand and contains indices of the Pauli Zs in the product
        e.g. if terms = [(0.5, (0,1)), (0.7, (0,1,2,3)), (-2, (1,2))]
        the Hamiltonian is 0.5*Z0Z1 + 0.7*Z0Z1Z2Z3 - 2*Z1Z2
    """
    rng = np.random.RandomState(seed)
    random.seed(seed)
    num_edges = rng.poisson(d * N)
    indices = random.choices(list(combinations(range(N), q)), k=num_edges)
    all_terms = []
    for term in indices:
        all_terms.append((rng.choice([-1 / np.sqrt(d), 1 / np.sqrt(d)]), term))
    return all_terms
