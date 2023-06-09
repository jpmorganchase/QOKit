###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import pandas as pd
import numpy as np
import json

df = pd.read_json("../qokit/assets/best_known_QAOA_parameters_wrt_MF.json", orient="index")

N_min: int = 22
p_max: int = 150

outpath_mean = f"../qokit/assets/mean_params_0320_MF.json"
outpath_median = f"../qokit/assets/median_params_0320_MF.json"

donor_params_mean = {}
donor_params_median = {}

for p in range(1, p_max + 1):
    dftmp = df[(df["p"] == p) & (df["N"] >= N_min)]
    Ns = sorted(set(dftmp["N"]))

    donor_params_mean[p] = {
        "beta": list(np.mean([np.array(dftmp[dftmp["N"] == N].squeeze()["beta"]) for N in Ns], axis=0)),
        "gamma": list(
            np.mean(
                [np.array(dftmp[dftmp["N"] == N].squeeze()["gamma"]) * N for N in Ns],
                axis=0,
            )
        ),
    }
    donor_params_median[p] = {
        "beta": list(np.median([np.array(dftmp[dftmp["N"] == N].squeeze()["beta"]) for N in Ns], axis=0)),
        "gamma": list(
            np.median(
                [np.array(dftmp[dftmp["N"] == N].squeeze()["gamma"]) * N for N in Ns],
                axis=0,
            )
        ),
    }

json.dump(donor_params_mean, open(outpath_mean, "w"))
json.dump(donor_params_median, open(outpath_median, "w"))
