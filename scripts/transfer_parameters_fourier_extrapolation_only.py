###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import os

os.environ["OMP_NUM_THREADS"] = "90"
os.environ["NUMBA_NUM_THREADS"] = "90"

import numpy as np
import pickle
from pathlib import Path
import time
import sys
import json

sys.path.append("../code/")

from qaoa_objective_labs import get_qaoa_labs_objective
from parameter_utils import (
    from_fourier_basis,
    to_fourier_basis,
    extrapolate_parameters_in_fourier_basis,
)


def transfer_for_N(gamma_donor, beta_donor, N, p_start, p_end, step_size, p_donor):
    beta_next = np.array(beta_donor)
    gamma_next = np.array(gamma_donor) / N

    for p in range(p_start, p_end, step_size):
        start = time.time()
        outpath = f"data/0228_extrapolate_mean_overlap_{N}_{p}_{p_donor}.pickle"
        if Path(outpath).exists():
            print(f"Found precomputed pickle at {outpath}, skipping", flush=True)
            row = pickle.load(open(outpath, "rb"))
            beta_next = row["beta"]
            gamma_next = row["gamma"]
            continue
        f = get_qaoa_labs_objective(N, p, parameterization="gamma beta", objective="expectation and overlap")
        u, v = to_fourier_basis(gamma_next, beta_next)
        beta_next, gamma_next = from_fourier_basis(*extrapolate_parameters_in_fourier_basis(u, v, p, step_size))

        e, o = f(gamma_next, beta_next)
        row = {
            "overlap transferred": 1 - o,
            "merit factor transferred": -e,
            "N": N,
            "p": p,
            "beta": beta_next,
            "gamma": gamma_next,
            "p_donor": p_donor,
        }
        end = time.time()
        print(f"{N}\t{p}\t{row['overlap transferred']}\t{row['merit factor transferred']:.3f}\t{end-start:.2f} sec")
        pickle.dump(row, open(outpath, "wb"))


step_size = 1
p_donor = 26

params_dict = json.load(open("../qokit/assets/mean_params_0228.json", "r"))[str(p_donor)]
donor_params = {
    "beta": np.array(params_dict["beta"]),
    "gamma": np.array(params_dict["gamma"]),
}

for N in range(20, 34):
    transfer_for_N(
        donor_params["gamma"],
        donor_params["beta"],
        N,
        p_donor + step_size,
        50,
        step_size,
        p_donor,
    )
