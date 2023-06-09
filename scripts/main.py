###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import os

os.environ["OMP_NUM_THREADS"] = "1"

import pathos

mp = pathos.helpers.mp
import numpy as np
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

import nlopt


def minimize_nlopt(f, X0, rhobeg=None, p=None):
    def nlopt_wrapper(x, grad):
        if grad.size > 0:
            sys.exit("Shouldn't be calling a gradient!")
        return f(x)

    opt = nlopt.opt(nlopt.LN_BOBYQA, 2 * p)
    opt.set_min_objective(nlopt_wrapper)

    opt.set_xtol_rel(1e-8)
    opt.set_ftol_rel(1e-8)
    opt.set_initial_step(rhobeg)

    xstar = opt.optimize(X0)
    minf = opt.last_optimum_value()

    return xstar, minf


def optimize_for_N_p(N, p, terms, offset, seed_for_initial_points=1):
    outpath = f"data/1119_init_small_{N}_{p}_{seed_for_initial_points}.pickle"
    if Path(outpath).exists():
        print(f"Found precomputed pickle at {outpath}, skipping", flush=True)
        return

    f = get_qaoa_labs_objective(N, p, terms, offset)

    nseeds = 19 * p
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
    minimize = partial(minimize_nlopt, rhobeg=rhobeg, p=p)

    # if N < 10 and nseeds > 80:
    #     nprocesses = 80
    # else:
    #     nprocesses = 24
    nprocesses = 95

    # If the following fails, the hotfix is downgrading dill:
    # pip install dill==0.3.5.1
    # from https://github.com/uqfoundation/dill/issues/332
    with mp.Pool(nprocesses) as pool:
        all_res = pool.starmap(
            minimize,
            [(f, np.hstack([gamma, beta])) for gamma, beta in zip(init_gammas, init_betas)],
        )

    best_f = float("inf")
    best_x = None
    for xstar, minf in all_res:
        if minf < best_f:
            best_x = copy.deepcopy(xstar)
            best_f = minf

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
    print(
        "\n"
        + "!" * 100
        + "\nBe warned! nprocesses and OMP_NUM_THREADS must be chosen by hand! This version is optimal for small N and 96vCPU\n"
        + "!" * 100
        + "\n"
    )
    for N in [15]:
        terms, offset = get_energy_term_indices(N)

        for p in range(11, 40):
            optimize_for_N_p(N, p, terms, offset)
