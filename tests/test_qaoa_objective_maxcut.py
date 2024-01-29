###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from functools import partial
from qiskit.providers.aer import AerSimulator
import pytest

from qokit.maxcut import maxcut_obj, get_adjacency_matrix, get_maxcut_terms

from qokit.qaoa_objective_maxcut import get_qaoa_maxcut_objective

from qokit.qaoa_circuit_maxcut import get_qaoa_circuit, get_parameterized_qaoa_circuit
from qokit.utils import brute_force, precompute_energies
from qokit.parameter_utils import get_sk_gamma_beta, get_fixed_gamma_beta
import qokit
from qokit.fur import get_available_simulators, get_available_simulator_names

test_maxcut_folder = Path(__file__).parent


qiskit_backend = AerSimulator(method="statevector")
SIMULATORS = get_available_simulators("x") + get_available_simulators("xyring") + get_available_simulators("xycomplete")
simulators_to_run_names = get_available_simulator_names("x") + ["qiskit"]
simulators_to_run_names_no_qiskit = get_available_simulator_names("x")


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


@pytest.mark.parametrize("simulator", simulators_to_run_names)
def test_maxcut_qaoa_obj_fixed_angles(simulator):
    N = 8
    for d, max_p in [(3, 11), (5, 4)]:
        G = nx.random_regular_graph(d, N)

        obj = partial(maxcut_obj, w=get_adjacency_matrix(G))
        optimal_cut, x = brute_force(obj, N, function_takes="bits")
        for simulator in simulators_to_run_names:
            last_overlap = 0
            for p in range(1, max_p + 1):
                gamma, beta, AR = get_fixed_gamma_beta(d, p, return_AR=True)
                f_e = get_qaoa_maxcut_objective(N, p, G=G, parameterization="gamma beta", simulator=simulator, objective="expectation")
                f_o = get_qaoa_maxcut_objective(N, p, G=G, parameterization="gamma beta", simulator=simulator, objective="overlap")
                assert -1 * f_e(gamma, beta) / optimal_cut > AR
                current_overlap = 1 - f_o(gamma, beta)
                if current_overlap < 0.5:
                    # high values of overlap are unreliable
                    assert current_overlap > last_overlap
                last_overlap = current_overlap


def test_maxcut_qaoa_obj_consistency_across_simulators():
    N = 8
    for d, p in [(3, 11), (5, 4)]:
        G = nx.random_regular_graph(d, N)
        gamma, beta = get_fixed_gamma_beta(d, p)
        for objective in ["expectation", "overlap"]:
            qaoa_objectives = [
                get_qaoa_maxcut_objective(N, p, G=G, parameterization="gamma beta", simulator=simulator, objective=objective)(gamma, beta)
                for simulator in simulators_to_run_names
            ]
            assert np.all(np.isclose(qaoa_objectives, qaoa_objectives[0]))


@pytest.mark.parametrize("simulator", simulators_to_run_names_no_qiskit)
def test_maxcut_qaoa_obj_fixed_angles_with_terms_and_precomputed_energies(simulator):
    N = 10
    for d, max_p in [(3, 11), (5, 4)]:
        G = nx.random_regular_graph(d, N)
        obj = partial(maxcut_obj, w=get_adjacency_matrix(G))
        precomputed_energies = precompute_energies(obj, N)
        optimal_cut, x = brute_force(obj, N, function_takes="bits")
        for p in range(1, max_p + 1):
            gamma, beta, AR = get_fixed_gamma_beta(d, p, return_AR=True)
            f1 = get_qaoa_maxcut_objective(N, p, G=G, parameterization="gamma beta", simulator=simulator)
            f2 = get_qaoa_maxcut_objective(N, p, precomputed_cuts=precomputed_energies, parameterization="gamma beta", simulator=simulator)
            e1 = f1(gamma, beta)
            e2 = f2(gamma, beta)
            assert -1 * e1 / optimal_cut > AR
            assert -1 * e2 / optimal_cut > AR
            assert np.isclose(e1, e2)


@pytest.mark.parametrize("simulator", simulators_to_run_names)
def test_maxcut_weighted_qaoa_obj(simulator):
    # The dataframe is a sample from '../qokit/assets/maxcut_datasets/weighted_Shaydulin_Lotshaw_2022.json'
    df = pd.read_json(Path(test_maxcut_folder, "sample_from_weighted_Shaydulin_Lotshaw_2022.json"), orient="index")

    df["G"] = df.apply(
        lambda row: nx.node_link_graph(row["G_json"]),
        axis=1,
    )
    for _, row in df.iterrows():
        f = get_qaoa_maxcut_objective(row["G"].number_of_nodes(), row["p"], G=row["G"], parameterization="gamma beta", simulator=simulator)
        assert np.isclose(-f(row["gamma"], row["beta"]), row["Expected cut of QAOA"])


def test_maxcut_weighted_qaoa_obj_qiskit_circuit():
    # Qiskit non-parameterized circuit must be tested separately
    # The dataframe is a sample from '../qokit/assets/maxcut_datasets/weighted_Shaydulin_Lotshaw_2022.json'
    df = pd.read_json(Path(test_maxcut_folder, "sample_from_weighted_Shaydulin_Lotshaw_2022.json"), orient="index")

    df["G"] = df.apply(
        lambda row: nx.node_link_graph(row["G_json"]),
        axis=1,
    )
    for _, row in df.iterrows():
        precomputed_cuts = precompute_energies(maxcut_obj, row["G"].number_of_nodes(), w=get_adjacency_matrix(row["G"]))
        qc = get_qaoa_circuit(row["G"], row["beta"], row["gamma"])
        qc_param = get_parameterized_qaoa_circuit(row["G"], row["p"]).bind_parameters(np.hstack([row["beta"], row["gamma"]]))
        sv = np.asarray(qiskit_backend.run(qc).result().get_statevector())
        sv_param = np.asarray(qiskit_backend.run(qc_param).result().get_statevector())

        assert np.allclose(sv, sv_param)
        assert np.isclose(precomputed_cuts.dot(np.abs(sv) ** 2), row["Expected cut of QAOA"])


@pytest.mark.parametrize("simclass", SIMULATORS)
def test_maxcut_precompute(simclass):
    N = 4
    G = nx.random_regular_graph(3, N)
    print(G.edges())
    for u, v, w in G.edges(data=True):
        w["weight"] = np.random.rand()
    precomputed_cuts = precompute_energies(maxcut_obj, N, w=get_adjacency_matrix(G))
    terms = get_maxcut_terms(G)
    sim = simclass(N, terms=terms)
    cuts = sim.get_cost_diagonal()
    assert np.allclose(precomputed_cuts, cuts, atol=1e-6)


@pytest.mark.parametrize("simulator", simulators_to_run_names)
def test_sk_ini_maxcut(simulator):
    N = 10
    for d, max_p in [(3, 5), (5, 5)]:
        G = nx.random_regular_graph(d, N)
        obj = partial(maxcut_obj, w=get_adjacency_matrix(G))
        optimal_cut, x = brute_force(obj, N, function_takes="bits")
        precomputed_energies = precompute_energies(obj, N)
        last_ar = 0
        for p in range(1, max_p + 1):
            gamma, beta = get_sk_gamma_beta(p)
            f = get_qaoa_maxcut_objective(N, p, G=G, parameterization="gamma beta", simulator=simulator)
            cur_ar = -f(gamma / np.sqrt(d), beta) / optimal_cut
            if p == 1:
                assert cur_ar > np.mean(precomputed_energies) / optimal_cut
            else:
                assert cur_ar > last_ar
                last_ar = cur_ar
