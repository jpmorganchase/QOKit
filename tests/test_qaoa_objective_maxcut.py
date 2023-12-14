###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import networkx as nx
import numpy as np
from qokit.qaoa_objective_maxcut import get_qaoa_maxcut_objective
from qokit.utils import precompute_energies
from qokit.parameter_utils import get_fixed_gamma_beta
from qokit.maxcut import maxcut_obj, get_adjacency_matrix


def test_validate_energy_for_terms_and_precomputedcuts_are_same():
    G = nx.random_regular_graph(3, 16)
    # without precomputed cuts
    f_terms = get_qaoa_maxcut_objective(G.number_of_nodes(), 1, G=G, parameterization="gamma beta")
    # with precomputed cuts
    precomputed_cuts = precompute_energies(maxcut_obj, G.number_of_nodes(), w=get_adjacency_matrix(G))
    f_precomputedcuts = get_qaoa_maxcut_objective(G.number_of_nodes(), 1, precomputed_cuts=precomputed_cuts, parameterization="gamma beta")
    p = 1
    gamma, beta = get_fixed_gamma_beta(3, p)
    energy_terms = f_terms(-1 * np.asarray(gamma), beta)
    energy_precomputedcuts = f_precomputedcuts(gamma, beta)
    assert energy_terms == energy_precomputedcuts


def test_validate_energy_for_terms_with_simulators_are_same():
    G = nx.random_regular_graph(3, 16)
    f = f = get_qaoa_maxcut_objective(G.number_of_nodes(), 1, G=G, parameterization="gamma beta", simulator="auto")
    g = get_qaoa_maxcut_objective(G.number_of_nodes(), 1, G=G, parameterization="gamma beta", simulator="qiskit")
    p = 1
    gamma, beta = get_fixed_gamma_beta(3, p)
    auto = f(gamma, beta)
    qiskit = g(gamma, beta)
    assert np.isclose(auto, qiskit)
