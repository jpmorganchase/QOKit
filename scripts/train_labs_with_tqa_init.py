###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
"""Train the QAOA for LABS problem with trotterized quantum annealing approach.
    We train train the QAOA with the cost function being the overlap wrt the ground state. 
    We set the cutoff of the overlap at 0.5
"""
import os

os.environ["OMP_NUM_THREADS"] = "4"

import pathos

mp = pathos.helpers.mp
import numpy as np
from scipy.optimize import minimize
import copy
import pickle
from pathlib import Path

import sys

sys.path.append("../code/")

from labs import (
    get_energy_term_indices,
)
from qaoa_objective_labs import get_qaoa_labs_overlap


def optimize_for_N_p(N, p, terms, offset):
    """Return:
    Overlap of QAOA(p) optimized state with the ground state.
    Parameters
    ----------
    N : int
        Number of qubits
    p : int
        Number of QAOA layers (number of parameters will be 2*p)
    terms : list of tuples
            Containing indices of the Pauli Zs in the product
    offset : int
            energy offset required due to constant factors (identity terms)
            not included in the Hamiltonian
    Returns
    -------
    overlap : float
              Overlap of QAOA(p) optimized state with the ground state.
    """
    outpath = f"data_tqa/tqa_overlap_{N}_{p}.pickle"
    if Path(outpath).exists():
        print(f"Found precomputed pickle at {outpath}, skipping", flush=True)
        return

    f = get_qaoa_labs_overlap(N, p, terms, offset)

    # define TQA initialization

    delta_t = 0.75
    gamma_init = np.zeros(p)
    beta_init = np.zeros(p)
    theta_init = np.zeros(2 * p)
    for i in range(p):
        gamma_init[i] = (i + 1) * delta_t / p
        beta_init[i] = (1 - ((i + 1) / p)) * delta_t
    theta_init = np.hstack([gamma_init, beta_init])

    # minimize_cobyla = partial(minimize, method="COBYLA", rhobeg=0.005)

    # if N < 10 and nseeds > 80:
    #    nprocesses = 80
    # else:
    #    nprocesses = 24

    # with mp.Pool(nprocesses) as pool:
    #    all_res = pool.starmap(minimize, [(f, np.hstack([gamma, beta])) for gamma, beta in zip(init_gammas, init_betas)])

    res = minimize(f, theta_init, method="COBYLA", options={"rhobeg": 0.005})

    best_f = 1 - res.fun
    best_x = copy.deepcopy(res.x)

    row = {
        "N": N,
        "p": p,
        "overlap": best_f,
        "theta": best_x,
        "terms": terms,
        "gamma": best_x[:p],
        "beta": best_x[p:],
    }
    print(f"Found overlap={row['overlap']:.3f} at p={p}, saving to {outpath}")
    pickle.dump(row, open(outpath, "wb"))
    return best_f


if __name__ == "__main__":
    for N in range(8, 15):
        terms, offset = get_energy_term_indices(N)

        p = 1
        while p < 30:
            overlap = optimize_for_N_p(N, p, terms, offset)
            if overlap >= 0.5:
                break
            p += 1
