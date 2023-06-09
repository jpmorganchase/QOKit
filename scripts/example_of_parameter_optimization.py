###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import numpy as np
from qiskit.providers.aer import AerSimulator
from scipy.optimize import minimize

import os
import sys

sys.path.append("../code/")

from qaoa_circuit_labs import get_parameterized_qaoa_circuit
from labs import (
    get_energy_term_indices,
    negative_merit_factor_from_bitstring_faster,
)
from utils import precompute_energies

starting_point_dir = "./starting_points/"
results_dir = "./benchmark_results/"

os.makedirs(starting_point_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)


def optimize_for_N_p(N, p, terms, precomputed_energies):
    global true_F
    qc_param = get_parameterized_qaoa_circuit(N, terms, p)

    def f(theta):
        global true_F
        p = len(theta) // 2
        gamma = theta[:p]
        beta = theta[p:]
        qc = qc_param.bind_parameters(np.hstack([beta, gamma]))
        backend = AerSimulator(method="statevector")
        sv = np.asarray(backend.run(qc).result().get_statevector())
        amplitudes = np.array([np.abs(sv[kk]) ** 2 for kk in range(sv.shape[0])])
        f_theta = precomputed_energies.dot(amplitudes)

        # Storing objective values for benchmarking optimization methods
        if len(true_F):
            true_F = np.vstack((true_F, f_theta))
        else:
            true_F = np.array([f_theta])

        return f_theta

    seeds = range(10)
    solvers = ["COBYLA"]
    delta = 0.01
    nfmax = int(150)  # Max number of evaluations to be used by optimizer

    for seed in seeds:
        res_filename = os.path.join(results_dir, "results_seed=" + str(seed) + "_nfmax=" + str(nfmax) + ".npy")
        starting_point_filename = os.path.join(starting_point_dir, "seed=" + str(seed) + ".csv")

        if os.path.exists(res_filename):
            continue

        Res = {}

        X0 = np.random.uniform(0, np.pi, 2 * p)  # starting parameters for the optimizer
        if not os.path.exists(res_filename):
            np.savetxt(starting_point_filename, X0, delimiter=",")

        for solver in solvers:
            np.random.seed(seed)
            true_F = np.empty(0)

            if solver == "COBYLA":
                obj = lambda theta: f(theta)

                res = minimize(
                    obj,
                    X0,
                    method="COBYLA",
                    options={"maxiter": nfmax, "rhobeg": delta},
                )

            Res[solver] = true_F

        np.save(res_filename, Res)


if __name__ == "__main__":
    N = 7
    p = 2
    terms, offset = get_energy_term_indices(N)
    precomputed_energies = precompute_energies(negative_merit_factor_from_bitstring_faster, N, terms, offset)
    optimize_for_N_p(N, p, terms, precomputed_energies)
