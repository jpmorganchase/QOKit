###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
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
    true_optimal_mf,
)
from qaoa_objective_labs import get_qaoa_labs_objective


def optimize_for_N_p(N, p, terms, offset, seed_for_initial_points=1):
    outpath = f"data/1119_init_small_{N}_{p}_{seed_for_initial_points}.pickle"
    if Path(outpath).exists():
        print(f"Found precomputed pickle at {outpath}, skipping", flush=True)
        return

    f = get_qaoa_labs_objective(N, p, terms, offset)

    nseeds = 20 * p
    np.random.seed(seed_for_initial_points)
    init_betas = np.random.uniform(-0.2, -0.1, (nseeds, p))
    init_gammas = np.random.uniform(0, 0.85 / N, (nseeds, p))

    if N <= 7:
        nseed_unif = 100 * p
        unif_betas = np.random.uniform(0, np.pi, (nseed_unif, p))
        unif_gammas = np.random.uniform(0, np.pi, (nseed_unif, p))

        init_betas = np.vstack((init_betas, unif_betas))
        init_gammas = np.vstack((init_gammas, unif_gammas))

    rhobeg = 0.01 / N
    minimize_cobyla = partial(minimize, method="COBYLA", rhobeg=rhobeg)

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
            best_f = res.fun

    row = {
        "N": N,
        "p": p,
        "AR": -best_f / true_optimal_mf[N],
        "merit factor": -best_f,
        "nseeds": nseeds,
        "theta": best_x,
        "terms": terms,
        "gamma": best_x[:p],
        "beta": best_x[p:],
    }
    print(f"Found MF={row['merit factor']:.3f} at p={p}, optimal {true_optimal_mf[N]} saving to {outpath}")
    pickle.dump(row, open(outpath, "wb"))


if __name__ == "__main__":
    for N in range(8, 20):
        terms, offset = get_energy_term_indices(N)

        for p in range(1, 30):
            optimize_for_N_p(N, p, terms, offset)
