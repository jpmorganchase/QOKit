###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
"""Train the QAOA for LABS problem with cold start technique
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
from functools import partial

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
    outpath = f"data_cold/cold_overlap_stop_{N}_{p}.pickle"
    if Path(outpath).exists():
        print(f"Found precomputed pickle at {outpath}, skipping", flush=True)
        return

    f = get_qaoa_labs_overlap(N, p, terms, offset)

    nseeds = 20 * p
    init_betas = np.random.uniform(-0.2, -0.1, (nseeds, p))
    init_gammas = np.random.uniform(0, 0.85 / N, (nseeds, p))

    if N <= 7:
        nseed_unif = 100 * p
        unif_betas = np.random.uniform(0, np.pi, (nseed_unif, p))
        unif_gammas = np.random.uniform(0, np.pi, (nseed_unif, p))

        init_betas = np.vstack((init_betas, unif_betas))
        init_gammas = np.vstack((init_gammas, unif_gammas))

    minimize_cobyla = partial(minimize, method="COBYLA", rhobeg=0.005)

    if N < 10 and nseeds > 80:
        nprocesses = 80
    else:
        nprocesses = 24

    with mp.Pool(nprocesses) as pool:
        all_res = pool.starmap(
            minimize,
            [(f, np.hstack([gamma, beta])) for gamma, beta in zip(init_gammas, init_betas)],
        )

    best_f = float("inf")
    best_x = None
    for res in all_res:
        if res.fun < best_f:
            best_x = copy.deepcopy(res.x)
            best_f = 1 - res.fun

    row = {
        "N": N,
        "p": p,
        "overlap": best_f,
        "nseeds": nseeds,
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
