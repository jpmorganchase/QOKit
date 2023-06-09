###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
"""Train the QAOA for LABS problem with Fourier based initialization as provided in https://arxiv.org/pdf/1812.01041.pdf.
    We train train the QAOA with the cost function being the overlap wrt the ground state. 
    We set the cutoff of the overlap at 0.3
"""
import os

os.environ["OMP_NUM_THREADS"] = "32"
os.environ["NUMBA_NUM_THREADS"] = "32"

import numpy as np
import pandas as pd
import copy
import pickle
from pathlib import Path
from functools import partial
from itertools import starmap
import argparse
import sys

sys.path.append("../code/")

from labs import (
    get_energy_term_indices,
)
from qaoa_objective_labs import get_qaoa_labs_objective
from utils import get_all_best_known
from parameter_utils import (
    from_fourier_basis,
    to_fourier_basis,
    extrapolate_parameters_in_fourier_basis,
)

import nlopt

# Get the precomputed dataframe from the json

df = get_all_best_known()


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


def optimize_for_N_1(N, p, terms, objective_to_optimize):
    """Return:
    Overlap of QAOA(p=1) optimized state with the ground state.

    Parameters
    ----------
    N : int
        Number of qubits
    p : int
        Number of QAOA layers (number of parameters will be 2*p)
    terms : list of tuples
            Containing indices of the Pauli Zs in the product

    Returns
    -------
    overlap : float
              Overlap of QAOA(p) optimized state with the ground state.
    u : list
        Optimized frequencies of gamma paramter for QAOA(p)
    v : list
        Optimized frequencies of beta paramter for QAOA(p)
    """
    if objective_to_optimize == "expectation":
        suffix = " opt4MF"
        label = "merit factor"
        beta_min = -0.2
        beta_max = -0.1
        gamma_min = 0.0
        gamma_max = 0.85 / N
    else:
        suffix = " opt4overlap"
        label = "overlap"
        beta_min = -0.3
        beta_max = -0.15
        gamma_min = 0.6 / N
        gamma_max = 1.2 / N

    df_slice = df[(df["N"] == N) & (df["p"] == p)]
    if len(df_slice) > 0:
        row = df_slice.squeeze()
        if pd.notna(row[label + suffix]):
            print(
                f"Found precomputed result, not recomputing for N={N}, p={p}",
                flush=True,
            )
            best_f = row[label + suffix]
            best_x = np.hstack([row["gamma" + suffix], row["beta" + suffix]])
            # extract final u and v from the optimised theta params
            u, v = to_fourier_basis(best_x[:p], best_x[p:])
            return best_f, u, v

    outpath = f"data/fourier_MF_{N}_{p}.pickle"
    print(f"Using {outpath} to save future result")
    if Path(outpath).exists():
        print(f"Found precomputed pickle at {outpath}, skipping", flush=True)
        row = pickle.load(open(outpath, "rb"))
        best_f = row[label]
        best_x = np.hstack([row["gamma"], row["beta"]])
        u, v = to_fourier_basis(best_x[:p], best_x[p:])
        return best_f, u, v

    f = get_qaoa_labs_objective(N, p, objective=objective_to_optimize)
    nseeds = 400
    init_betas = np.random.uniform(beta_min, beta_max, (nseeds, p))
    init_gammas = np.random.uniform(gamma_min, gamma_max, (nseeds, p))

    rhobeg = 0.01 / N
    minimize = partial(minimize_nlopt, rhobeg=rhobeg, p=p)

    all_res = starmap(
        minimize,
        [(f, np.hstack([gamma, beta])) for gamma, beta in zip(init_gammas, init_betas)],
    )

    best_f = float("inf")
    best_x = None
    for xstar, minf in all_res:
        if minf < best_f:
            best_x = copy.deepcopy(xstar)
            best_f = minf

    if objective_to_optimize == "expectation":
        best_f = -best_f
    else:
        best_f = 1 - best_f

    row = {
        "N": N,
        "p": p,
        label: best_f,
        "nseeds": nseeds,
        "terms": terms,
        "gamma": best_x[:p],
        "beta": best_x[p:],
    }
    print(f"Found {objective_to_optimize}={row[label]:.3f} at p={p}, saving to {outpath}")
    pickle.dump(row, open(outpath, "wb"))
    # extract final u and v from the optimised theta params
    u, v = to_fourier_basis(best_x[:p], best_x[p:])
    return best_f, u, v


def optimize_for_N_p(N, p, terms, step_size, u, v, objective_to_optimize):
    """Return:
    Overlap of QAOA(p > 1) optimized state with the ground state.

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
    step_size : int
                number of jumps one wants to perform in p : p - step_size -> p

    u : list
        Optimized frequencies of gamma paramter for QAOA(p - step_size)
    v : list
        Optimized frequencies of beta paramter for QAOA(p - step_size)

    Returns
    -------
    overlap : float
              Overlap of QAOA(p) optimized state with the ground state.
    u : list
        Optimized frequencies of gamma paramter for QAOA(p)
    v : list
        Optimized frequencies of beta paramter for QAOA(p)
    """

    if objective_to_optimize == "expectation":
        suffix = " opt4MF"
        label = "merit factor"
    else:
        suffix = " opt4overlap"
        label = "overlap"

    df_slice = df[(df["N"] == N) & (df["p"] == p)]
    if len(df_slice) > 0:
        row = df_slice.squeeze()
        if pd.notna(row[label + suffix]):
            print(
                f"Found precomputed result, not recomputing for N={N}, p={p}",
                flush=True,
            )
            best_f = row[label + suffix]
            best_x = np.hstack([row["gamma" + suffix], row["beta" + suffix]])
            # extract final u and v from the optimised theta params
            u, v = to_fourier_basis(best_x[:p], best_x[p:])
            return best_f, u, v

    outpath = f"data/fourier_MF_{N}_{p}.pickle"
    print(f"Using {outpath} to save future result")
    if Path(outpath).exists():
        print(f"Found precomputed pickle at {outpath}, skipping", flush=True)
        row = pickle.load(open(outpath, "rb"))
        best_f = row[label]
        best_x = np.hstack([row["gamma"], row["beta"]])
        u, v = to_fourier_basis(best_x[:p], best_x[p:])
        return best_f, u, v

    # only if not precomputed result exists, do the optimization
    f = get_qaoa_labs_objective(N, p, objective=objective_to_optimize, parameterization="freq")
    init_u, init_v = extrapolate_parameters_in_fourier_basis(u, v, p, step_size)
    init_freq = np.hstack([init_u, init_v])

    rhobeg = 0.01 / N
    xstar, minf = minimize_nlopt(f, init_freq, rhobeg=rhobeg, p=p)

    if objective_to_optimize == "expectation":
        label = "merit factor"
        best_f = -minf
    else:
        label = "overlap"
        best_f = 1 - minf
    best_x = copy.deepcopy(xstar)
    u = best_x[:p]
    v = best_x[p:]
    beta, gamma = from_fourier_basis(u, v)
    row = {
        "N": N,
        "p": p,
        label: best_f,
        "terms": terms,
        "gamma": gamma,
        "beta": beta,
    }
    print(f"Found {objective_to_optimize}={row[label]:.3f} at p={p}, saving to {outpath}")
    pickle.dump(row, open(outpath, "wb"))
    return best_f, u, v


def run_fourier_for_N(N):
    objective_to_optimize = "expectation"
    # objective_to_optimize = "overlap"

    terms, offset = get_energy_term_indices(N)

    if objective_to_optimize == "overlap":
        # Optimize for p = 1 layer
        overlap, u, v = optimize_for_N_1(N, 1, terms, objective_to_optimize)
        overlap_cutoff = 0.3
        step_size = 1
        max_p = 150
        p = 1 + step_size
        # Optimize for p >1 layers
        if overlap <= overlap_cutoff:
            while p <= max_p:
                overlap, u, v = optimize_for_N_p(N, p, terms, step_size, u, v, objective_to_optimize)
                if overlap >= overlap_cutoff:
                    break
                p += step_size
    else:
        # Optimize for p = 1 layer
        _, u, v = optimize_for_N_1(N, 1, terms, objective_to_optimize)
        step_size = 1
        max_p = 200
        p = 1 + step_size
        while p <= max_p:
            _, u, v = optimize_for_N_p(N, p, terms, step_size, u, v, objective_to_optimize)
            p += step_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("N", help="number of spins", type=int)
    args = parser.parse_args()

    run_fourier_for_N(args.N)
