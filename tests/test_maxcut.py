import numpy as np
import pandas as pd
import networkx as nx
import pytest
from pathlib import Path
from functools import partial
from qiskit.providers.aer import AerSimulator

from qokit.maxcut import maxcut_obj, get_adjacency_matrix

from qokit.qaoa_objective_maxcut import get_qaoa_maxcut_objective

from qokit.qaoa_circuit_maxcut import get_qaoa_circuit, get_parameterized_qaoa_circuit
from qokit.utils import brute_force, precompute_energies, get_fixed_gamma_beta

test_maxcut_folder = Path(__file__).parent


qiskit_backend = AerSimulator(method="statevector")


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


def test_maxcut_qaoa_obj_fixed_angles():
    N = 10
    for d, max_p in [(3, 11), (5, 4)]:
        G = nx.random_regular_graph(d, N)

        obj = partial(maxcut_obj, w=get_adjacency_matrix(G))
        optimal_cut, x = brute_force(obj, N, function_takes="bits")

        for p in range(1, max_p + 1):
            gamma, beta, AR = get_fixed_gamma_beta(d, p, return_AR=True)
            for simulator in ["auto", "qiskit"]:
                f = get_qaoa_maxcut_objective(N, p, G=G, parameterization="gamma beta", simulator=simulator)
                assert f(gamma, beta) / optimal_cut > AR


def test_maxcut_weighted_qaoa_obj():
    # The dataframe is a sample from '../qokit/assets/maxcut_datasets/weighted_Shaydulin_Lotshaw_2022.json'
    df = pd.read_json(Path(test_maxcut_folder, "sample_from_weighted_Shaydulin_Lotshaw_2022.json"), orient="index")

    df["G"] = df.apply(
        lambda row: nx.node_link_graph(row["G_json"]),
        axis=1,
    )

    for _, row in df.iterrows():
        for simulator in ["auto", "qiskit"]:
            f = get_qaoa_maxcut_objective(row["G"].number_of_nodes(), row["p"], G=row["G"], parameterization="gamma beta", simulator=simulator)
            assert np.isclose(f(row["gamma"], row["beta"]), row["Expected cut of QAOA"])

        # Qiskit non-parameterized circuit must be tested separately
        precomputed_cuts = precompute_energies(maxcut_obj, row["G"].number_of_nodes(), w=get_adjacency_matrix(row["G"]))
        qc = get_qaoa_circuit(row["G"], row["beta"], row["gamma"])
        qc_param = get_parameterized_qaoa_circuit(row["G"], row["p"]).bind_parameters(np.hstack([row["beta"], row["gamma"]]))

        sv = np.asarray(qiskit_backend.run(qc).result().get_statevector())
        sv_param = np.asarray(qiskit_backend.run(qc_param).result().get_statevector())

        assert np.allclose(sv, sv_param)
        assert np.isclose(precomputed_cuts.dot(np.abs(sv) ** 2), row["Expected cut of QAOA"])
