###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import os

os.environ["OMP_NUM_THREADS"] = "50"
os.environ["NUMBA_NUM_THREADS"] = "50"

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict
import time
import sys

sys.path.append("../qokit/")

from qaoa_objective_labs import get_qaoa_labs_objective

df = pd.read_json("../qokit/assets/best_known_QAOA_parameters_wrt_overlap.json", orient="index")

all_donor_params = defaultdict(dict)


def transfer_for_N_p(N_donor, N, p):
    start = time.time()
    outpath = f"data/023_transfer_overlap_{N}_{p}_{N_donor}.pickle"
    if Path(outpath).exists():
        print(f"Found precomputed pickle at {outpath}, skipping", flush=True)
        return
    donor_params = all_donor_params[N_donor]
    if p not in donor_params.keys():
        print(f"Skipping p={p}, N_donor={N_donor}")
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


all_N_donor = [30]

# Donor parameters: transferring from N=N_donor
for N_donor in all_N_donor:
    for _, row in df[df["N"] == N_donor].iterrows():
        all_donor_params[N_donor][row["p"]] = {
            "beta": row["beta"],
            "gamma": np.array(row["gamma"]) * N_donor,
        }  # using the scaling rule obtained in plot_heatmaps.ipynb

for N in range(20, 34):
    for p in range(1, 100):
        transfer_for_N_p(all_N_donor[0], N, p)
# with Pool(1) as pool:
#     pool.starmap(transfer_for_N_p, product(all_N_donor, [33], range(1, 100)))
