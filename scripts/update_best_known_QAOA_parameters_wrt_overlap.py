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

import sys

sys.path.append("../qokit/")
from objectives import get_qaoa_labs_objective, get_qaoa_labs_overlap

update_best_known_best_known_QAOA_parameters_folder = Path(__file__).parent
best_known_QAOA_parameters_path = Path(
    update_best_known_best_known_QAOA_parameters_folder,
    "../qokit/assets/best_known_QAOA_parameters_wrt_overlap.json",
)


def add_MF_to_row(row):
    if "merit factor" in row and pd.notna(row["merit factor"]):
        return row["merit factor"]
    else:
        return -get_qaoa_labs_objective(row["N"], row["p"], parameterization="gamma beta")(row["gamma"], row["beta"])


def update_best_known_QAOA_parameters_from_dataframe(df, test_values=True, column_to_maximize="overlap"):
    df_best_known = pd.read_json(best_known_QAOA_parameters_path, orient="index")
    # new DataFrame should have extra columns removed
    assert set(df.columns).issubset(set(df_best_known.columns))
    # Column over which we are updating the rows should be present
    assert column_to_maximize in df.columns
    assert column_to_maximize in df_best_known.columns

    if test_values:
        for _, row in tqdm(df.iterrows(), desc="Rows of the new DataFrame verified", total=len(df)):
            f = get_qaoa_labs_overlap(row["N"], row["p"])
            x = np.hstack((row["gamma"], row["beta"]))
            assert np.isclose(1 - f(x), row["overlap"])
        print("Verified the overlap: OK")

    dfnew = pd.concat([df, df_best_known])
    idx = dfnew.groupby(["N", "p"])[column_to_maximize].transform(max) == dfnew[column_to_maximize]
    # Sorting and resetting the index to minimize diff
    dfnew = dfnew[idx].sort_values(["N", "p"]).reset_index(drop=True)

    # add merit factor field if missing
    tqdm.pandas(desc="Adding missing merit factor fields to the dataframe")
    dfnew["merit factor"] = dfnew.progress_apply(add_MF_to_row, axis=1)
    dfnew.to_json(best_known_QAOA_parameters_path, indent=4, orient="index")


if __name__ == "__main__":
    import pickle

    rows = []
    for fname in Path("data_fourier/").glob("fourier_overlap_*"):
        rows.append(pickle.load(open(fname, "rb")))
    df = pd.DataFrame(rows, columns=rows[0].keys())
    df = df.drop(["terms"], axis=1)
    update_best_known_QAOA_parameters_from_dataframe(df, test_values=True)
