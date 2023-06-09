###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import os

os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
from qiskit.providers.aer import AerSimulator
from scipy.optimize import minimize_scalar
import pickle
from pathlib import Path
import multiprocessing
import time


import sys

sys.path.append("../code/")

from qaoa_circuit_labs import get_parameterized_qaoa_circuit
from labs import (
    get_energy_term_indices,
    negative_merit_factor_from_bitstring,
    true_optimal_mf,
)
from utils import precompute_energies, obj_from_statevector, get_ramp


def optimize_for_N_p(N, p, terms, precomputed_energies):
    outpath = f"data/1107_optimized_ramp_{N}_{p}.pickle"
    if Path(outpath).exists():
        print(f"Found precomputed pickle at {outpath}, skipping", flush=True)
        return

    start = time.time()
    qc_param = get_parameterized_qaoa_circuit(N, terms, p)

    def f(delta):
        ramp = get_ramp(delta, p)

        gamma = ramp["gamma"]
        beta = ramp["beta"]
        # qc = get_qaoa_circuit(N, terms, beta, gamma)
        qc = qc_param.bind_parameters(np.hstack([beta, gamma]))

        backend = AerSimulator(method="statevector")
        sv = backend.run(qc).result().get_statevector()

        return obj_from_statevector(
            sv,
            negative_merit_factor_from_bitstring,
            precomputed_energies=precomputed_energies,
        )

    res = minimize_scalar(f, bounds=[0, 0.4], method="bounded", options={"maxiter": 100, "xatol": 1e-4})

    end = time.time()
    total_time = end - start

    row = {
        "N": N,
        "p": p,
        "res": res,
        "merit factor": -res.fun,
        "delta": res.x,
        "total time": total_time,
        "nfev": res.nfev,
        "time per iteration": total_time / res.nfev,
    }
    print(
        f"Found MF={row['merit factor']:.3f} at p={p}, delta={row['delta']}, {res.nfev} circuits executed in {row['total time']:.2f}s ({row['time per iteration']:.2f}s/it), optimal {true_optimal_mf[N]}, saving to {outpath}"
    )
    pickle.dump(row, open(outpath, "wb"))


if __name__ == "__main__":
    N = 5
    terms, offset = get_energy_term_indices(N)
    precomputed_energies = precompute_energies(negative_merit_factor_from_bitstring, N)

    min_p = 5
    step_size_p = 5
    # for p in np.arange(min_p, 100000, step_size_p):
    #     optimize_for_N_p(N, p, terms, precomputed_energies)
    with multiprocessing.Pool(4) as p:
        p.starmap(
            optimize_for_N_p,
            [(N, x, terms, precomputed_energies) for x in np.arange(min_p, 50, step_size_p)],
        )
