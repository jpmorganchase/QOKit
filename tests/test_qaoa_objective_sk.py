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

from qokit.sk import sk_obj, get_adjacency_matrix, get_sk_terms

from qokit.qaoa_objective_sk import get_qaoa_sk_objective

from qokit.qaoa_circuit_sk import get_qaoa_circuit, get_parameterized_qaoa_circuit
from qokit.utils import brute_force, precompute_energies
from qokit.parameter_utils import get_sk_gamma_beta, get_fixed_gamma_beta
import qokit
from qokit.fur import get_available_simulators, get_available_simulator_names

test_sk_folder = Path(__file__).parent


qiskit_backend = AerSimulator(method="statevector")
SIMULATORS = get_available_simulators("x") + get_available_simulators("xyring") + get_available_simulators("xycomplete")
simulators_to_run_names = get_available_simulator_names("x") + ["qiskit"]
simulators_to_run_names_no_qiskit = get_available_simulator_names("x")


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

@pytest.mark.skip
@pytest.mark.parametrize("simulator", simulators_to_run_names)
def test_sk_qaoa_obj_fixed_angles(simulator):
    N = 8
    G = nx.complete_graph(N)
    J = np.random.randn(N, N)
    J = (J + J.T)/2

    for edge in G.edges:
        G.edges[edge[0], edge[1]]['weight'] = J[edge[0], edge[1]]

    for d, max_p in [(N-1, 11), (N-1, 4)]:
        obj = partial(sk_obj, w=get_adjacency_matrix(G))
        optimal_cut, x = brute_force(obj, N, function_takes="bits")
        for simulator in simulators_to_run_names:
            last_overlap = 0
            for p in range(1, max_p + 1):
                gamma, beta, AR = get_fixed_gamma_beta(d, p, return_AR=True)
                f_e = get_qaoa_sk_objective(N, p, G=G, parameterization="gamma beta", simulator=simulator, objective="expectation")
                f_o = get_qaoa_sk_objective(N, p, G=G, parameterization="gamma beta", simulator=simulator, objective="overlap")
                assert -1 * f_e(gamma, beta) / optimal_cut > AR
                current_overlap = 1 - f_o(gamma, beta)
                if current_overlap < 0.5:
                    # high values of overlap are unreliable
                    assert current_overlap > last_overlap
                last_overlap = current_overlap


def test_sk_qaoa_obj_consistency_across_simulators():
    N = 8
    G = nx.complete_graph(N)
    J = np.random.randn(N, N)
    J = (J + J.T)/2

    for edge in G.edges:
        G.edges[edge[0], edge[1]]['weight'] = J[edge[0], edge[1]]

    for d, p in [(N-1, 11), (N-1, 4)]:
        gamma, beta = get_fixed_gamma_beta(d, p)
        for objective in ["expectation", "overlap"]:
            qaoa_objectives = [
                get_qaoa_sk_objective(N, p, G=G, parameterization="gamma beta", simulator=simulator, objective=objective)(gamma, beta)
                for simulator in simulators_to_run_names
            ]
            assert np.all(np.isclose(qaoa_objectives, qaoa_objectives[0]))


@pytest.mark.skip
@pytest.mark.parametrize("simulator", simulators_to_run_names_no_qiskit)
def test_sk_qaoa_obj_fixed_angles_with_terms_and_precomputed_energies(simulator):
    N = 10
    for d, max_p in [(3, 11), (5, 4)]:
        G = nx.random_regular_graph(d, N)
        obj = partial(sk_obj, w=get_adjacency_matrix(G))
        precomputed_energies = precompute_energies(obj, N)
        optimal_cut, x = brute_force(obj, N, function_takes="bits")
        for p in range(1, max_p + 1):
            gamma, beta, AR = get_fixed_gamma_beta(d, p, return_AR=True)
            f1 = get_qaoa_sk_objective(N, p, G=G, parameterization="gamma beta", simulator=simulator)
            f2 = get_qaoa_sk_objective(N, p, precomputed_cuts=precomputed_energies, parameterization="gamma beta", simulator=simulator)
            e1 = f1(gamma, beta)
            e2 = f2(gamma, beta)
            assert -1 * e1 / optimal_cut > AR
            assert -1 * e2 / optimal_cut > AR
            assert np.isclose(e1, e2)


@pytest.mark.skip
@pytest.mark.parametrize("simulator", simulators_to_run_names)
def test_sk_weighted_qaoa_obj(simulator):
    # The dataframe is a sample from '../qokit/assets/maxcut_datasets/weighted_Shaydulin_Lotshaw_2022.json'
    df = pd.read_json(Path(test_sk_folder, "sample_from_weighted_Shaydulin_Lotshaw_2022.json"), orient="index")

    df["G"] = df.apply(
        lambda row: nx.node_link_graph(row["G_json"]),
        axis=1,
    )
    for _, row in df.iterrows():
        f = get_qaoa_sk_objective(row["G"].number_of_nodes(), row["p"], G=row["G"], parameterization="gamma beta", simulator=simulator)
        assert np.isclose(-f(row["gamma"], row["beta"]), row["Expected cut of QAOA"])


@pytest.mark.skip
def test_sk_weighted_qaoa_obj_qiskit_circuit():
    # Qiskit non-parameterized circuit must be tested separately
    # The dataframe is a sample from '../qokit/assets/maxcut_datasets/weighted_Shaydulin_Lotshaw_2022.json'
    df = pd.read_json(Path(test_sk_folder, "sample_from_weighted_Shaydulin_Lotshaw_2022.json"), orient="index")

    df["G"] = df.apply(
        lambda row: nx.node_link_graph(row["G_json"]),
        axis=1,
    )
    for _, row in df.iterrows():
        precomputed_cuts = precompute_energies(sk_obj, row["G"].number_of_nodes(), w=get_adjacency_matrix(row["G"]))
        qc = get_qaoa_circuit(row["G"], row["gamma"], row["beta"])
        qc_param = get_parameterized_qaoa_circuit(row["G"], row["p"]).assign_parameters(np.hstack([row["beta"], row["gamma"]]))
        sv = np.asarray(qiskit_backend.run(qc).result().get_statevector())
        sv_param = np.asarray(qiskit_backend.run(qc_param).result().get_statevector())

        assert np.allclose(sv, sv_param)
        assert np.isclose(precomputed_cuts.dot(np.abs(sv) ** 2), row["Expected cut of QAOA"])


@pytest.mark.parametrize("simclass", SIMULATORS)
def test_sk_precompute(simclass):
    N = 4
    G = nx.complete_graph(N)
    J = np.random.randn(N, N)
    J = (J + J.T)/2

    for edge in G.edges:
        G.edges[edge[0], edge[1]]['weight'] = J[edge[0], edge[1]]

    print(G.edges())
    for u, v, w in G.edges(data=True):
        w["weight"] = np.random.rand()
    precomputed_cuts = precompute_energies(sk_obj, N, w=get_adjacency_matrix(G))
    terms = get_sk_terms(G)
    sim = simclass(N, terms=terms)
    cuts = sim.get_cost_diagonal()
    assert np.allclose(precomputed_cuts, cuts, atol=1e-6)


@pytest.mark.parametrize("simulator", simulators_to_run_names)
def test_sk_ini_sk(simulator):
    N = 10
    G = nx.complete_graph(N)
    J = np.random.randn(N, N)
    J = (J + J.T)/2
    for edge in G.edges:
        G.edges[edge[0], edge[1]]['weight'] = J[edge[0], edge[1]]
    for d, max_p in [(3, 5), (5, 5)]:
        obj = partial(sk_obj, w=get_adjacency_matrix(G))
        optimal_cut, x = brute_force(obj, N, function_takes="bits")
        precomputed_energies = precompute_energies(obj, N)
        last_ar = 0
        for p in range(1, max_p + 1):
            gamma, beta = get_sk_gamma_beta(p)
            f = get_qaoa_sk_objective(N, p, G=G, parameterization="gamma beta", simulator=simulator)
            cur_ar = -f(gamma / np.sqrt(d), beta) / optimal_cut
            if p == 1:
                assert cur_ar > np.mean(precomputed_energies) / optimal_cut
            else:
                assert cur_ar > last_ar
                last_ar = cur_ar

@pytest.mark.parametrize("simulator", simulators_to_run_names_no_qiskit)
def test_overlap_sk(simulator):
    N = 4
    J = np.random.randn(N, N)
    J = (J + J.T)/2
    G = nx.complete_graph(N)
    for edge in G.edges:
        G.edges[edge[0], edge[1]]['weight'] = J[edge[0], edge[1]]
    p = 1
    beta = [np.random.uniform(0, 1)]
    gamma = [np.random.uniform(0, 1)]

    obj = partial(sk_obj, w=get_adjacency_matrix(G))
    precomputed_energies = precompute_energies(obj, N)

    f1 = get_qaoa_sk_objective(N, p, precomputed_cuts=precomputed_energies, parameterization="gamma beta", objective="overlap")
    f2 = get_qaoa_sk_objective(N, p, G=G, parameterization="gamma beta", objective="overlap")

    print("f1:", f1(gamma, beta))
    print("f2:", f2(gamma, beta))
    assert np.isclose(f1(gamma, beta), f2(gamma, beta))
    assert np.isclose(f1([0], [0]), f2([0], [0]))

    maxval = precomputed_energies.max()
    bitstring_loc = (precomputed_energies == maxval).nonzero()
    assert len(bitstring_loc) == 1
    bitstring_loc = bitstring_loc[0]
    assert np.isclose(1 - f1([0], [0]), len(bitstring_loc) / len(precomputed_energies))
