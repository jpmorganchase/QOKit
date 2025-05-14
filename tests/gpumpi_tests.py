import pandas as pd
import numpy as np
import networkx as nx
from functools import partial
from qokit.maxcut import maxcut_obj, get_adjacency_matrix, get_maxcut_terms
from qokit.qaoa_objective_labs import get_qaoa_labs_objective
from qokit.qaoa_objective_maxcut import get_qaoa_maxcut_objective
from qokit.sk import sk_obj, get_random_J
from qokit.qaoa_objective_sk import get_qaoa_sk_objective
from qokit.utils import brute_force, precompute_energies
from qokit.labs import get_terms
import subprocess
from qokit.fur.mpi_nbcuda.qaoa_simulator import mpi_available
from qokit.fur import get_available_simulator_names, choose_simulator


def gpumpi_test(N=16, p=4, seed=1):
    df = pd.read_json("./qokit/assets/QAOA_with_fixed_parameters_p_opt.json", orient="index")
    row = df[(df["N"] == N) & (df["p"] == p)]
    beta = row["beta"].values[0]
    gamma = row["gamma"].values[0]
    overlap_trans_labs = float(row["overlap transferred"].values[0])
    f_gpumpi_labs = get_qaoa_labs_objective(N, p, simulator="gpumpi", objective="overlap", parameterization="gamma beta")
    overlap_trans_computed_labs = 1 - f_gpumpi_labs(gamma, beta)

    print(np.isclose(overlap_trans_computed_labs, overlap_trans_labs))

    # Maxcut
    d = 3
    G = nx.random_regular_graph(d, N, seed=seed)
    obj_maxcut = partial(maxcut_obj, w=get_adjacency_matrix(G))
    precomputed_energies = precompute_energies(obj_maxcut, N)

    o1_c_maxcut = get_qaoa_maxcut_objective(N, p, precomputed_cuts=precomputed_energies, simulator="c", parameterization="gamma beta", objective="overlap")(
        gamma, beta
    )
    o1_gpumpi_maxcut = get_qaoa_maxcut_objective(
        N, p, precomputed_cuts=precomputed_energies, simulator="gpumpi", parameterization="gamma beta", objective="overlap"
    )(gamma, beta)
    o2_gpumpi_maxcut = get_qaoa_maxcut_objective(N, p, G=G, simulator="gpumpi", parameterization="gamma beta", objective="overlap")(gamma, beta)
    print(np.all([o1_c_maxcut == np.real(o1_gpumpi_maxcut), o1_c_maxcut == np.real(o2_gpumpi_maxcut)]))

    # SK
    J = get_random_J(N=N)
    obj_sk = partial(sk_obj, J=J)
    precomputed_energies = precompute_energies(obj_sk, N)

    o1_c_sk = get_qaoa_sk_objective(N, p, J=J, precomputed_energies=precomputed_energies, simulator="c", parameterization="gamma beta", objective="overlap")(
        gamma, beta
    )
    o1_gpumpi_sk = get_qaoa_sk_objective(
        N, p, J=J, precomputed_energies=precomputed_energies, simulator="gpumpi", parameterization="gamma beta", objective="overlap"
    )(gamma, beta)
    o2_gpumpi_sk = get_qaoa_sk_objective(N, p, J=J, simulator="gpumpi", parameterization="gamma beta", objective="overlap")(gamma, beta)

    print(np.all([np.isclose(o1_c_sk, np.real(o1_gpumpi_sk)), np.isclose(o1_c_sk, np.real(o2_gpumpi_sk))]))


if __name__ == "__main__":

    gpumpi_test()
