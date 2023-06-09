###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import os

os.environ["OMP_NUM_THREADS"] = "45"
os.environ["NUMBA_NUM_THREADS"] = "45"

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import time
import sys

sys.path.append("../qokit/")

from qaoa_objective_labs import get_qaoa_labs_objective

df = pd.read_json("../qokit/assets/best_known_QAOA_parameters_wrt_overlap.json", orient="index")
df_donor = pd.read_json("../qokit/assets/parameters_batch_0113.json", orient="index")

all_donor_params = dict()


def transfer_for_N_p(N, p):
    start = time.time()
    outpath = f"data/0113_transfer_from_batch_{N}_{p}.pickle"
    if Path(outpath).exists():
        print(f"Found precomputed pickle at {outpath}, skipping", flush=True)
        return
    if p not in all_donor_params.keys():
        print(f"Skipping p={p}")
        return
    beta = all_donor_params[p]["beta"]
    gamma = all_donor_params[p]["gamma"] / N

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


all_ps = []

# Donor parameters: transferring from N=N_donor
for _, row in df_donor.iterrows():
    all_ps.append(row["p"])
    all_donor_params[row["p"]] = {
        "beta": row["beta"],
        "gamma": np.array(row["gamma"]),
    }  # using the scaling rule obtained in plot_heatmaps.ipynb

for N in [34, 35]:
    for p in all_ps:
        transfer_for_N_p(N, p)

# parallel version
# with Pool(2) as pool:
#     pool.starmap(transfer_for_N_p, product(range(13, 32), all_ps))

# N = int(sys.argv[1])
# N_donor = all_N_donor[0]
# print(f"Transferring from {N_donor} to {N}")
# transfer_for_N_p(N_donor, N, p_donor)
