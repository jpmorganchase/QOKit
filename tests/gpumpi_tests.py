import pandas as pd
import numpy as np
from qokit.qaoa_objective_labs import get_qaoa_labs_objective
from qokit.labs import get_terms
import subprocess
from qokit.fur.mpi_nbcuda.qaoa_simulator import mpi_available
from qokit.fur import get_available_simulator_names, choose_simulator


def gpumpi_test(N=22, p=4):
    df = pd.read_json("./qokit/assets/QAOA_with_fixed_parameters_p_opt.json", orient="index")
    row = df[(df["N"] == N) & (df["p"] == p)]
    beta = row["beta"].values[0]
    gamma = row["gamma"].values[0]
    overlap_trans = float(row["overlap transferred"].values[0])

    terms = get_terms(N)
    simclass = choose_simulator(name="gpumpi")
    mysim = simclass(N, terms=terms)

    f_gpumpi = get_qaoa_labs_objective(N, p, simulator="gpumpi", objective="overlap", parameterization="gamma beta")

    overlap_trans_computed = 1 - f_gpumpi(gamma, beta)

    print(np.isclose(overlap_trans_computed, overlap_trans))


if __name__ == "__main__":

    gpumpi_test()
