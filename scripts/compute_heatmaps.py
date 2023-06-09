###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
# Compute heatmaps for p=1 QAOA
import os


os.environ["OMP_NUM_THREADS"] = "4"

import pathos

mp = pathos.helpers.mp

import numpy as np
from itertools import product
import pickle
from pathlib import Path

import sys

sys.path.append("../code/")
from qaoa_objective_labs import get_qaoa_labs_objective


def get_heatmap_for_N(
    N: int,
    min_beta: float = 0,
    max_beta: float = np.pi,
    min_gamma: float = 0,
    max_gamma=np.pi,
    npoints: int = 100,
    nprocesses: int = 16,
):
    """
    To plot (note that the result of the function is transposed):
        ```
        plt.imshow(
            -res['hm'],
            extent=[res['min_gamma'], res['max_gamma'],res['max_beta'], res['min_beta']]
        )
        plt.ylabel(r"$\beta$")
        plt.xlabel(r"$\gamma$")
        ```
    """
    # objective_to_optimize = "expectation"
    objective_to_optimize = "overlap"
    f = get_qaoa_labs_objective(N, 1, parameterization="gamma beta", objective=objective_to_optimize)

    betas = np.linspace(min_beta, max_beta, npoints)
    gammas = np.linspace(min_gamma, max_gamma, npoints)

    with mp.Pool(nprocesses) as p:
        hm = p.starmap(f, product(gammas, betas))

    return np.array(hm).reshape(npoints, npoints).T


if __name__ == "__main__":
    for N in range(8, 25):
        outpath = f"data_overlap/heatmap_small_{N}.pickle"
        if Path(outpath).exists():
            print(f"Found heatmap at {outpath}, skipping")
            continue

        res = {
            "min_beta": -0.5,
            "max_beta": 0,
            "min_gamma": 0,
            "max_gamma": 0.25,
        }
        res["hm"] = get_heatmap_for_N(
            N,
            min_beta=res["min_beta"],
            max_beta=res["max_beta"],
            min_gamma=res["min_gamma"],
            max_gamma=res["max_gamma"],
        )
        print(f"Done computing heatmap, saving to {outpath}")
        pickle.dump(res, open(outpath, "wb"))
