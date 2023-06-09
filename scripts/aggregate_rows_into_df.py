###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
"""
Aggregates and updates (if possible) transfer results

Takes as input rows produced by transfer_parameters.py
"""

import pandas as pd
from pathlib import Path
import pickle

update_transfer_results_folder = Path(__file__).parent

# dict mapping from the wildcard of transfer_parameters to
wildcard_to_prettyname = {
    # "1205_transfer_*_22.pickle": "transferred_from_22_MF.json",
    # "1124_extrapolate_fourier_*_22.pickle": "transferred_from_22_with_fourier_extrapolation_MF.json",
    # "0113_transfer_from_batch_*.pickle": "transferred_from_batch_0113.json",
    # "0123_transfer_from_batch_*.pickle": "transferred_from_batch_0123.json",
    # "0124_transfer_from_batch_*.pickle": "transferred_from_batch_0124.json",
    # "fourier_batch_overlap_unscaled_18_23_*.pickle": "parameters_batch_0113.json",
    # "fourier_batch_1124_17_24_*.pickle": "parameters_batch_0124.json",
    # "0215_extrapolate_0124_batch_*.pickle": "transferred_extrapolated_from_batch_0124_at_p_10.json",
    # "0217_transfer_overlap_median_*.pickle": "transferred_mean_overlap_0217.json", # yes, there was a typo in the label
    # "0217_transfer_gibbs_mean_*.pickle": "transferred_mean_gibbs_0217.json",
    "0223_extrapolate_mean_overlap_*.pickle": "transferred_extrapolated_from_mean_overlap_0217_at_p_16.json",
    "0227_extrapolate_mean_overlap_*.pickle": "transferred_extrapolated_from_mean_overlap_0227_at_p_23.json",
    "0228_extrapolate_mean_overlap_*.pickle": "transferred_extrapolated_from_mean_overlap_0228.json",
    "0227_transfer_overlap_mean_*.pickle": "transferred_mean_overlap_0227.json",
    "0228_transfer_overlap_mean_*.pickle": "transferred_mean_overlap_0228.json",
    "../polaris_transfer_batch/data/02-21_0113_transfer_from_batch_*.pickle": "transferred_Dan_0221.json",
    "../polaris_transfer_batch/data/02-26_0113_transfer_from_batch_*.pickle": "transferred_Dan_0226.json",
    "../polaris_transfer_batch/data/03-01_0113_transfer_from_batch_*.pickle": "transferred_Dan_0301.json",
    "../polaris_transfer_batch/data/0228_transfer_from_batch_*.pickle": "transferred_Dan_mean_0228.json",
    "0320_transfer_MF_mean_*": "trasferred_mean_MF_0320.json",
}


def update_dataframe(wildcard):
    df_path = Path(update_transfer_results_folder, "../qokit/assets/", wildcard_to_prettyname[wildcard])
    if df_path.exists():
        df_old = pd.read_json(df_path, orient="index")
    else:
        df_old = None
    newrows = []
    for fname in Path("data/").glob(wildcard):
        row = pickle.load(open(fname, "rb"))
        if (
            (df_old is None)
            or ("N" in row and len(df_old[(df_old["N"] == row["N"]) & (df_old["p"] == row["p"])]) == 0)
            or ("N" not in row and len(df_old[df_old["p"] == row["p"]]) == 0)
        ):
            newrows.append(row)

    if len(newrows) == 0:
        print(f"No new rows for {wildcard}, exiting")
        return
    else:
        print(f"Adding {len(newrows)} new rows for {wildcard}")

    df_with_new_rows = pd.DataFrame(newrows, columns=newrows[0].keys())

    if df_old is None:
        df_updated = df_with_new_rows
    else:
        df_updated = pd.concat([df_old, df_with_new_rows])

    # Sorting and resetting the index to minimize diff
    if "N" in df_updated.columns:
        df_updated = df_updated.sort_values(["N", "p"]).reset_index(drop=True)
    else:
        df_updated = df_updated.sort_values(["p"]).reset_index(drop=True)
    df_updated.to_json(df_path, indent=4, orient="index")


if __name__ == "__main__":
    for wildcard in wildcard_to_prettyname:
        update_dataframe(wildcard)
