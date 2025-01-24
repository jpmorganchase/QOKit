###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from functools import partial
from qiskit_aer import AerSimulator
import pytest

from qokit.sk import sk_obj, get_sk_terms

from qokit.maxcut import maxcut_obj, get_adjacency_matrix, get_maxcut_terms

from qokit.qaoa_objective_sk import get_qaoa_sk_objective
from qokit.qaoa_objective_maxcut import get_qaoa_maxcut_objective

from qokit.qaoa_circuit_sk import get_qaoa_circuit, get_parameterized_qaoa_circuit
from qokit.utils import brute_force, precompute_energies
from qokit.parameter_utils import get_sk_gamma_beta, get_fixed_gamma_beta
import qokit
from qokit.fur import get_available_simulators, get_available_simulator_names

from qokit.qaoa_objective import get_qaoa_objective

test_sk_folder = Path(__file__).parent


qiskit_backend = AerSimulator(method="statevector")
SIMULATORS = get_available_simulators("x") + get_available_simulators("xyring") + get_available_simulators("xycomplete")
simulators_to_run_names = get_available_simulator_names("x") + ["qiskit"]
simulators_to_run_names_no_qiskit = get_available_simulator_names("x")


def test_sk_obj(n=5):
    J = np.random.randn(n, n)
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0)

    def sk_obj_simple(x, J):
        obj = 0
        for i in range(n):
            for j in range(i, n):
                obj += J[i, j] * (2 * x[i] - 1) * (2 * x[j] - 1)
        return -2 * obj / np.sqrt(n)

    x = np.random.choice([0, 1], n)
    assert np.isclose(sk_obj(x, J), sk_obj_simple(x, J))


def test_sk_qaoa_obj_consistency_across_simulators():
    N = 8
    G = nx.complete_graph(N)
    J = np.random.randn(N, N)
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0)

    for edge in G.edges:
        G.edges[edge[0], edge[1]]["weight"] = J[edge[0], edge[1]]

    for p in [11, 4]:
        gamma, beta = get_sk_gamma_beta(p)
        for objective in ["expectation", "overlap"]:
            qaoa_objectives = [
                get_qaoa_sk_objective(N, p, J=J, parameterization="gamma beta", simulator=simulator, objective=objective)(gamma, beta)
                for simulator in simulators_to_run_names_no_qiskit
            ]
            assert np.all(np.isclose(qaoa_objectives, qaoa_objectives[0]))


@pytest.mark.parametrize("simulator", simulators_to_run_names_no_qiskit)
def test_sk_qaoa_obj_fixed_angles_and_precomputed_energies(simulator):
    N = 10
    J = np.random.randn(N, N)
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0)
    for max_p in [11, 4]:
        obj = partial(sk_obj, J=J)
        precomputed_energies = precompute_energies(obj, N)
        for p in range(1, max_p + 1):
            gamma, beta = get_sk_gamma_beta(p)
            f1 = get_qaoa_sk_objective(N, p, J=J, parameterization="gamma beta", simulator=simulator)
            f2 = get_qaoa_sk_objective(N, p, J=J, precomputed_cuts=precomputed_energies, parameterization="gamma beta", simulator=simulator)
            e1 = f1(gamma, beta)
            e2 = f2(gamma, beta)
            assert np.isclose(e1, e2)


@pytest.mark.parametrize("simclass", SIMULATORS)
def test_sk_precompute(simclass):
    N = 4
    J = np.random.randn(N, N)
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0)

    precomputed_cuts = precompute_energies(sk_obj, N, J)
    terms = get_sk_terms(J)
    sim = simclass(N, terms=terms)
    cuts = sim.get_cost_diagonal()
    assert np.allclose(precomputed_cuts, cuts, atol=1e-6)


@pytest.mark.parametrize("simulator", simulators_to_run_names_no_qiskit)
def test_sk_maxcut_bruteforce(simulator):
    N = 10
    J = np.random.randn(N, N)
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0)

    obj_maxcut = partial(maxcut_obj, w=J)
    optimal_maxcut, x_maxcut = brute_force(obj_maxcut, num_variables=N, function_takes="bits")
    obj_sk = partial(sk_obj, J=J)
    optimal_sk, x_sk = brute_force(obj_sk, num_variables=N, function_takes="bits")
    x_maxcut_comp = 1 - x_maxcut

    assert np.allclose(x_maxcut, x_sk) or np.allclose(x_maxcut_comp, x_sk)


@pytest.mark.parametrize("simulator", simulators_to_run_names_no_qiskit)
def test_overlap_sk(simulator):
    N = 4
    J = np.random.randn(N, N)
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0)
    p = 1
    beta = [np.random.uniform(0, 1)]
    gamma = [np.random.uniform(0, 1)]

    obj = partial(sk_obj, J=J)
    precomputed_energies = precompute_energies(obj, N)

    f1 = get_qaoa_sk_objective(N, p, J=J, precomputed_cuts=precomputed_energies, parameterization="gamma beta", objective="overlap")
    f2 = get_qaoa_sk_objective(N, p, J=J, parameterization="gamma beta", objective="overlap")

    assert np.isclose(f1(gamma, beta), f2(gamma, beta))
    assert np.isclose(f1([0], [0]), f2([0], [0]))

    maxval = precomputed_energies.max()
    bitstring_loc = (precomputed_energies == maxval).nonzero()
    assert len(bitstring_loc) == 1
    bitstring_loc = bitstring_loc[0]
    assert np.isclose(1 - f1([0], [0]), len(bitstring_loc) / len(precomputed_energies))
