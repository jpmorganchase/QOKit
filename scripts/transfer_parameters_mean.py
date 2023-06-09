###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
"""
Parameters are obtained using compute_mean_median_parameters.py
"""

import os

os.environ["OMP_NUM_THREADS"] = "45"
os.environ["NUMBA_NUM_THREADS"] = "45"

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict
import time
import sys
import json

sys.path.append("../qokit/")

from qaoa_objective_labs import get_qaoa_labs_objective

df = pd.read_json("../qokit/assets/best_known_QAOA_parameters_wrt_MF.json", orient="index")

donor_params = defaultdict(dict)


def transfer_for_N_p(N, p):
    start = time.time()
    outpath = f"data/0320_transfer_MF_mean_{N}_{p}.pickle"
    if Path(outpath).exists():
        print(f"Found precomputed pickle at {outpath}, skipping", flush=True)
        return
    if p not in donor_params.keys():
        print(f"Skipping p={p}")
        return
    beta = donor_params[p]["beta"]
    gamma = donor_params[p]["gamma"] / N

    f = get_qaoa_labs_objective(N, p, parameterization="gamma beta", objective="expectation and overlap")
    e, o = f(gamma, beta)
    row = {
        "overlap transferred": 1 - o,
        "merit factor transferred": -e,
        "N": N,
        "p": p,
        "beta": beta,
        "gamma": gamma,
    }
    end = time.time()

    print(f"{N}\t{p}\t{row['overlap transferred']}\t{row['merit factor transferred']:.3f}\t{end-start:.2f} sec")

    this_row_in_df = df[(df["N"] == N) & (df["p"] == p)]
    if len(this_row_in_df) == 1:
        this_row_in_df = this_row_in_df.squeeze()
        row["merit factor"] = this_row_in_df["merit factor"]
        row["MF transf / MF optimal"] = row["merit factor transferred"] / row["merit factor"]
        print(f"MF transf / MF optimal:\t{row['MF transf / MF optimal']:.3f}")
        row["overlap"] = this_row_in_df["overlap"]
        row["overlap transf / overlap optimal"] = row["overlap transferred"] / row["overlap"]
        print(f"overlap transf / overlap optimal:\t{row['overlap transf / overlap optimal']:.3f}")

    pickle.dump(row, open(outpath, "wb"))


donor_params_json = json.load(open("../qokit/assets/mean_params_0320_MF.json", "r"))
for p, params_dict in donor_params_json.items():
    donor_params[int(p)] = {
        "beta": np.array(params_dict["beta"]),
        "gamma": np.array(params_dict["gamma"]),
    }

# for N in [33]: # range(20,33):
for N in range(20, 34):
    for p in range(1, max(donor_params.keys()) + 1):
        transfer_for_N_p(N, p)
