import pandas as pd
import numpy as np
import networkx as nx
from functools import partial
from qokit.maxcut import maxcut_obj, get_adjacency_matrix
from qokit.qaoa_objective_labs import get_qaoa_labs_objective
from qokit.qaoa_objective_maxcut import get_qaoa_maxcut_objective
from qokit.sk import sk_obj, get_random_J
from qokit.qaoa_objective_sk import get_qaoa_sk_objective
from qokit.utils import precompute_energies
from qokit.parameter_utils import get_sk_gamma_beta, get_fixed_gamma_beta
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def gpumpi_labs_test(N=10, p=4):
    df = pd.read_json(f"{SCRIPT_DIR.parent}/qokit/assets/QAOA_with_fixed_parameters_p_opt.json", orient="index")
    row = df[(df["N"] == N) & (df["p"] == p)]
    beta = row["beta"].values[0]
    gamma = row["gamma"].values[0]
    overlap_trans_labs = float(row["overlap transferred"].values[0])
    for objective in ["overlap", "expectation"]:
        f_gpumpi_labs = get_qaoa_labs_objective(N, p, simulator="gpumpi", objective=objective, parameterization="gamma beta")
        f_c_labs = get_qaoa_labs_objective(N, p, simulator="c", objective=objective, parameterization="gamma beta")
        if objective == "overlap":
            overlap_trans_gpumpi_computed_labs = 1 - f_gpumpi_labs(gamma, beta)
            overlap_trans_c_computed_labs = 1 - f_c_labs(gamma, beta)
            assert np.isclose(overlap_trans_gpumpi_computed_labs, overlap_trans_labs)
            assert np.isclose(overlap_trans_gpumpi_computed_labs, overlap_trans_c_computed_labs)
        elif objective == "expectation":
            assert np.isclose(f_gpumpi_labs(gamma, beta), f_c_labs(gamma, beta))


def gpumpi_maxcut_test(N=12, p=3, seed=1):
    d = 3
    G = nx.random_regular_graph(d, N, seed=seed)
    gamma, beta = get_fixed_gamma_beta(d, p)
    obj_maxcut = partial(maxcut_obj, w=get_adjacency_matrix(G))
    precomputed_energies = precompute_energies(obj_maxcut, N)
    for objective in ["overlap", "expectation"]:
        o1_c_maxcut = get_qaoa_maxcut_objective(N, p, precomputed_cuts=precomputed_energies, simulator="c", parameterization="gamma beta", objective=objective)(
            gamma, beta
        )
        o1_gpumpi_maxcut = get_qaoa_maxcut_objective(
            N, p, precomputed_cuts=precomputed_energies, simulator="gpumpi", parameterization="gamma beta", objective=objective
        )(gamma, beta)
        o2_gpumpi_maxcut = get_qaoa_maxcut_objective(N, p, G=G, simulator="gpumpi", parameterization="gamma beta", objective=objective)(gamma, beta)

        assert np.isclose(o1_c_maxcut, o1_gpumpi_maxcut.real)
        assert np.isclose(o1_c_maxcut, o2_gpumpi_maxcut.real)


def gpumpi_sk_test(N=10, p=3, seed=42):

    J = get_random_J(N=N, seed=seed)
    gamma, beta = get_sk_gamma_beta(p)
    obj_sk = partial(sk_obj, J=J)
    precomputed_energies = precompute_energies(obj_sk, N)
    for objective in ["overlap", "expectation"]:
        o1_c_sk = get_qaoa_sk_objective(
            N, p, J=J, precomputed_energies=precomputed_energies, simulator="c", parameterization="gamma beta", objective="overlap"
        )(gamma, beta)
        o1_gpumpi_sk = get_qaoa_sk_objective(
            N, p, J=J, precomputed_energies=precomputed_energies, simulator="gpumpi", parameterization="gamma beta", objective="overlap"
        )(gamma, beta)
        o2_gpumpi_sk = get_qaoa_sk_objective(N, p, J=J, simulator="gpumpi", parameterization="gamma beta", objective="overlap")(gamma, beta)

        assert np.isclose(o1_c_sk, o1_gpumpi_sk.real)
        assert np.isclose(o1_c_sk, o2_gpumpi_sk.real)


if __name__ == "__main__":

    gpumpi_labs_test()
    gpumpi_maxcut_test()
    gpumpi_sk_test()
