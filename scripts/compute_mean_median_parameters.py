###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import pandas as pd
import numpy as np
import json

df = pd.read_json("../qokit/assets/best_known_QAOA_parameters_wrt_overlap.json", orient="index")

for N_min, label in [(22, "0227"), (24, "0228")]:
    Ns = range(N_min, 32)
    p_max = df[df["N"] == Ns[-1]]["p"].max()

    outpath_mean = f"../qokit/assets/mean_params_{label}.json"
    outpath_median = f"../qokit/assets/median_params_{label}.json"

    donor_params_mean_old = json.load(open(outpath_mean, "r"))
    donor_params_median_old = json.load(open(outpath_median, "r"))

    donor_params_mean = {}
    donor_params_median = {}

    for p in range(1, p_max + 1):
        dftmp = df[df["p"] == p]
        donor_params_mean[p] = {
            "beta": list(
                np.mean(
                    [np.array(dftmp[dftmp["N"] == N].squeeze()["beta"]) for N in Ns],
                    axis=0,
                )
            ),
            "gamma": list(
                np.mean(
                    [np.array(dftmp[dftmp["N"] == N].squeeze()["gamma"]) * N for N in Ns],
                    axis=0,
                )
            ),
        }
        donor_params_median[p] = {
            "beta": list(
                np.median(
                    [np.array(dftmp[dftmp["N"] == N].squeeze()["beta"]) for N in Ns],
                    axis=0,
                )
            ),
            "gamma": list(
                np.median(
                    [np.array(dftmp[dftmp["N"] == N].squeeze()["gamma"]) * N for N in Ns],
                    axis=0,
                )
            ),
        }

    for d1, d2 in [
        (donor_params_median_old, donor_params_median),
        (donor_params_mean_old, donor_params_mean),
    ]:
        for k in d1.keys():
            for param in ["beta", "gamma"]:
                assert np.allclose(d1[k][param], d2[int(k)][param])

    json.dump(donor_params_mean, open(outpath_mean, "w"))
    json.dump(donor_params_median, open(outpath_median, "w"))
