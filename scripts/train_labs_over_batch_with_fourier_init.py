###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
"""Train the QAOA for LABS problem with Fourier based initialization as provided in https://arxiv.org/pdf/1812.01041.pdf.
"""

import os

os.environ["OMP_NUM_THREADS"] = "80"
os.environ["NUMBA_NUM_THREADS"] = "80"

import numpy as np
import copy
import pickle
from pathlib import Path
from functools import partial
from itertools import starmap
from tqdm import tqdm
import sys

sys.path.append("../code/")

from qaoa_objective_labs import get_qaoa_labs_objective
from parameter_utils import (
    from_fourier_basis,
    to_fourier_basis,
    extrapolate_parameters_in_fourier_basis,
)

import nlopt

# Get the precomputed dataframe from the json


def get_batch_loss_objective(N_min, N_max, p, **kwargs):
    """Inclusive of N_min and N_max
    kwargs are passed directly to get_qaoa_labs_objective
    """
    assert N_min <= N_max

    all_fs = []

    for N in range(N_min, N_max + 1):
        all_fs.append((N, get_qaoa_labs_objective(N, p, parameterization="gamma beta", **kwargs)))

    def batch_f(theta):
        beta, gamma = from_fourier_basis(theta[:p], theta[p:])
        # weighing the higher Ns more as overlap is expected to decay exponentially
        overlaps = np.array([(1 - f(gamma / N, beta)) * (1.7 ** (N - N_min)) for N, f in all_fs])
        # minus because we're minimizing, plus penalty on the second derivative
        return np.mean(-overlaps)

    return batch_f


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


def optimize_for_p_1(N_min, N_max, objective_to_optimize):
    p = 1
    if objective_to_optimize == "expectation":
        label = "merit factor"
        beta_min = -0.2
        beta_max = -0.1
        gamma_min = 0.0
        gamma_max = 0.85
    else:
        label = "overlap"
        beta_min = -0.3
        beta_max = -0.15
        gamma_min = 0.6
        gamma_max = 1.2

    outpath = f"data/fourier_batch_1124_{N_min}_{N_max}_{p}.pickle"
    print(f"Using {outpath} to save future result")
    if Path(outpath).exists():
        print(f"Found precomputed pickle at {outpath}, skipping", flush=True)
        row = pickle.load(open(outpath, "rb"))
        best_f = row[label]
        best_x = np.hstack([row["gamma"], row["beta"]])
        u, v = to_fourier_basis(best_x[:p], best_x[p:])
        return best_f, u, v

    f = get_batch_loss_objective(N_min, N_max, p, objective=objective_to_optimize)
    nseeds = 100
    init_betas = np.random.uniform(beta_min, beta_max, (nseeds, p))
    init_gammas = np.random.uniform(gamma_min, gamma_max, (nseeds, p))

    rhobeg = 0.01 / N_min
    minimize = partial(minimize_nlopt, rhobeg=rhobeg, p=p)

    all_res = tqdm(
        starmap(
            minimize,
            [(f, np.hstack([gamma, beta])) for gamma, beta in zip(init_gammas, init_betas)],
        ),
        total=nseeds,
        desc="Optimizing for p=1",
    )

    best_f = float("inf")
    best_x = None
    for xstar, minf in all_res:
        if minf < best_f:
            best_x = copy.deepcopy(xstar)
            best_f = minf

    row = {
        "N_min": N_min,
        "N_max": N_max,
        "p": p,
        label: -best_f,
        "nseeds": nseeds,
        "gamma": best_x[:p],
        "beta": best_x[p:],
    }
    print(f"Found {objective_to_optimize}={row[label]:.3f} at p={p}, saving to {outpath}")
    pickle.dump(row, open(outpath, "wb"))
    # extract final u and v from the optimised theta params
    u, v = to_fourier_basis(best_x[:p], best_x[p:])
    return best_f, u, v


def optimize_for_p(N_min, N_max, p, step_size, u, v, objective_to_optimize):
    if objective_to_optimize == "expectation":
        label = "merit factor"
    else:
        label = "overlap"

    outpath = f"data/fourier_batch_1124_{N_min}_{N_max}_{p}.pickle"
    print(f"Using {outpath} to save future result")
    if Path(outpath).exists():
        print(f"Found precomputed pickle at {outpath}, skipping", flush=True)
        row = pickle.load(open(outpath, "rb"))
        best_f = row[label]
        best_x = np.hstack([row["gamma"], row["beta"]])
        u, v = to_fourier_basis(best_x[:p], best_x[p:])
        return best_f, u, v

    # only if not precomputed result exists, do the optimization
    f = get_batch_loss_objective(N_min, N_max, p, objective=objective_to_optimize)
    init_u, init_v = extrapolate_parameters_in_fourier_basis(u, v, p, step_size)
    init_freq = np.hstack([init_u, init_v])

    rhobeg = 0.01 / N_min
    xstar, minf = minimize_nlopt(f, init_freq, rhobeg=rhobeg, p=p)

    if objective_to_optimize == "expectation":
        label = "merit factor"
    else:
        label = "overlap"
    best_f = -minf
    best_x = copy.deepcopy(xstar)
    u = best_x[:p]
    v = best_x[p:]
    beta, gamma = from_fourier_basis(u, v)
    row = {
        "N_min": N_min,
        "N_max": N_max,
        "p": p,
        label: best_f,
        "gamma": gamma,
        "beta": beta,
    }
    print(f"Found {objective_to_optimize}={row[label]:.3f} at p={p}, saving to {outpath}")
    pickle.dump(row, open(outpath, "wb"))
    return best_f, u, v


def run_fourier(N_min, N_max):
    objective_to_optimize = "overlap"

    # Optimize for p = 1 layer
    _, u, v = optimize_for_p_1(N_min, N_max, objective_to_optimize)
    step_size = 1
    max_p = 200
    p = 1 + step_size
    while p <= max_p:
        _, u, v = optimize_for_p(N_min, N_max, p, step_size, u, v, objective_to_optimize)
        p += step_size


if __name__ == "__main__":
    run_fourier(17, 24)
