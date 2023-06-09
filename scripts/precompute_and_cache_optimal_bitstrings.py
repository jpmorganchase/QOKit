###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
"""
Script used to precompute the optimal bitstrings in ../qokit/assets/precomputed_bitstrings/
"""
import sys

sys.path.append("../code/")

import time
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
from labs import (
    true_optimal_energy,
    energy_vals_from_bitstring,
)
from utils import yield_all_bitstrings


def is_optimal_bitstring(obj_f, x, optimal_f):
    f = obj_f(x)
    if np.isclose(f, optimal_f):
        return x
    else:
        return None


for N in [31, 32, 33, 34, 35]:
    outpath = f"../qokit/assets/precomputed_bitstrings/precomputed_bitstrings_{N}.npy"
    if Path(outpath).exists():
        print(f"Found precomputed numpy array in {outpath}, skipping")
        continue

    start = time.time()

    postfix_width = 10

    optimal_bitstrings = []
    for postfix_id in tqdm(range(2**postfix_width)):
        postfix = [int(x) for x in f"{postfix_id:0{postfix_width}b}"][::-1]
        bit_strings = (np.hstack([x, postfix]) for x in yield_all_bitstrings(N - len(postfix)))

        with Pool(30) as pool:
            optimal_bitstrings.extend(
                [
                    x
                    for x in pool.starmap(
                        is_optimal_bitstring,
                        (
                            (
                                energy_vals_from_bitstring,
                                bit_string,
                                true_optimal_energy[N],
                            )
                            for bit_string in bit_strings
                        ),
                    )
                    if x is not None
                ]
            )
    optimal_bitstrings = np.array(optimal_bitstrings)

    end = time.time()
    print(f"Done precomputing bitstrings for N={N} in {end-start:.2f}s, saving to {outpath}")
    np.save(outpath, optimal_bitstrings, allow_pickle=False)
