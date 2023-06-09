###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import os

os.environ["OMP_NUM_THREADS"] = "60"
os.environ["NUMBA_NUM_THREADS"] = "60"

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import time
import sys
from tqdm import tqdm

sys.path.append("../code/")

from qaoa_objective_labs import get_qaoa_labs_objective

df = pd.read_json("../qokit/assets/transferred_mean_overlap_0228.json", orient="index")


def obtain_overlaps(N, beta, gamma):
    start = time.time()

    p_max = len(beta)
    assert len(gamma) == p_max

    outpath = f"data/0302_intermediate_for_transferred_mean_overlap_0228_{N}_{p_max}.pickle"
    if Path(outpath).exists():
        print(f"Found precomputed pickle at {outpath}, skipping", flush=True)
        return

    overlaps = []
    merit_factors = []

    for p in tqdm(range(1, p_max + 1), desc=f"N={N}, p_max={p_max}"):
        e, o = get_qaoa_labs_objective(N, p, parameterization="gamma beta", objective="expectation and overlap")(gamma[:p], beta[:p])
        overlaps.append(1 - o)
        merit_factors.append(-e)

    row = {
        "overlaps": np.array(overlaps),
        "merit_factors": np.array(merit_factors),
        "N": N,
        "p_max": p_max,
        "beta": beta,
        "gamma": gamma,
    }
    end = time.time()

    print(f"{N}\t{p_max}\t{end-start:.2f} sec")

    pickle.dump(row, open(outpath, "wb"))


for p in [10, 20, 26]:
    for N in range(20, 34):
        row = df[(df["N"] == N) & (df["p"] == p)]
        if len(row) > 0:
            assert len(row) == 1
            row = row.squeeze()
            obtain_overlaps(N, row["beta"], row["gamma"])
