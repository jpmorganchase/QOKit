###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import os

os.environ["OMP_NUM_THREADS"] = "6"
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from functools import partial

import sys

sys.path.append("../qokit/")
from qaoa_objective_labs import get_qaoa_labs_objective, get_qaoa_labs_overlap
from labs import true_optimal_mf

update_best_known_best_known_QAOA_parameters_folder = Path(__file__).parent
best_known_QAOA_parameters_path = Path(
    update_best_known_best_known_QAOA_parameters_folder,
    "../qokit/assets/best_known_QAOA_parameters_wrt_MF.json",
)


def add_overlap_to_row(row, df_old=None):
    if "overlap" in row and pd.notna(row["overlap"]):
        return row["overlap"]
    else:
        # check if the old dataframe already has it
        df_slice = df_old[(df_old["N"] == row["N"]) & (df_old["p"] == row["p"])]
        if len(df_slice) > 0:
            row_old = df_slice.squeeze()
            if (
                "overlap" in row_old
                and pd.notna(row_old["overlap"])
                and np.allclose(row_old["gamma"], row["gamma"])
                and np.allclose(row_old["beta"], row["beta"])
            ):
                # reuse old overlap without recomputing
                # print(f"Reusing overlap from older row: {row_old['overlap']}")
                return row_old["overlap"]
        # print(f"Recomputing the overlap for N={row['N']}, p={row['p']}")
        return 1 - get_qaoa_labs_overlap(row["N"], row["p"], parameterization="gamma beta")(row["gamma"], row["beta"])


def update_best_known_QAOA_parameters_from_dataframe(df, test_values=True, column_to_maximize="AR"):
    df_best_known = pd.read_json(best_known_QAOA_parameters_path, orient="index")
    # new DataFrame should have extra columns removed
    assert set(df.columns).issubset(set(df_best_known.columns))
    # Column over which we are updating the rows should be present
    assert column_to_maximize in df.columns
    assert column_to_maximize in df_best_known.columns

    if test_values:
        for _, row in tqdm(df.iterrows(), desc="Rows of the new DataFrame verified", total=len(df)):
            f = get_qaoa_labs_objective(row["N"], row["p"])
            x = np.hstack((row["gamma"], row["beta"]))
            assert abs(f(x) + row["merit factor"]) <= 1e-10
        print("Verified the merit factors: OK")

    dfnew = pd.concat([df_best_known, df])
    idx = dfnew.groupby(["N", "p"])[column_to_maximize].transform(max) == dfnew[column_to_maximize]
    # Sorting and resetting the index to minimize diff
    dfnew = dfnew[idx].sort_values(["N", "p"]).reset_index(drop=True)

    # add overlap field if missing
    tqdm.pandas(desc="Adding missing overlap fields to the dataframe")
    add_overlap_partial = partial(add_overlap_to_row, df_old=df_best_known)
    dfnew["overlap"] = dfnew.progress_apply(add_overlap_partial, axis=1)
    dfnew.to_json(best_known_QAOA_parameters_path, indent=4, orient="index")


if __name__ == "__main__":
    import pickle

    # first, normal parameters
    rows = []
    for fname in Path("data/").glob("1119_init_small_*"):
        rows.append(pickle.load(open(fname, "rb")))
    df = pd.DataFrame(rows, columns=rows[0].keys())
    df = df.drop(["terms", "theta"], axis=1)
    update_best_known_QAOA_parameters_from_dataframe(df, test_values=False)

    # second, Fourier
    for fourier_stem in ["fourier_MF_*", "1213_fourier_MF_*"]:
        rows = []
        for fname in Path("data/").glob(fourier_stem):
            rows.append(pickle.load(open(fname, "rb")))
        df = pd.DataFrame(rows, columns=rows[0].keys())
        df = df.drop(["terms"], axis=1)
        df["AR"] = df.apply(lambda row: row["merit factor"] / true_optimal_mf[row["N"]], axis=1)
        update_best_known_QAOA_parameters_from_dataframe(df, test_values=False)
