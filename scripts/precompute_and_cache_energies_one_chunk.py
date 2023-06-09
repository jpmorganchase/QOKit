###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
"""
Script used to precompute the energies in ../qokit/assets/precomputed_energies/
"""

import sys

sys.path.append("../qokit/")

import time
import numpy as np
from pathlib import Path
import argparse
from labs import (
    negative_merit_factor_from_bitstring,
)
from utils import precompute_energies_parallel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("N", help="number of spins", type=int)
    parser.add_argument("postfix_id", help="postfix_id", type=int)
    parser.add_argument("postfix_width", help="width of the postfix", type=int)
    args = parser.parse_args()

    N = args.N
    postfix = [int(x) for x in f"{args.postfix_id:0{args.postfix_width}b}"][::-1]

    outpath = f"precomputed_energies_workspace/precomputed_energies_{N}_{args.postfix_id}.npy"
    if Path(outpath).exists():
        print(f"Found precomputed numpy array in {outpath}, skipping")
        sys.exit(0)

    print(f"Precomputing for N={N}, postfix={postfix}")
    start = time.time()
    precomputed_energies = precompute_energies_parallel(negative_merit_factor_from_bitstring, N, 30, postfix)
    end = time.time()
    print(f"Done precomputing chunk {args.postfix_id} for N={N} in {end-start:.2f}s, saving to {outpath}")
    np.save(outpath, precomputed_energies, allow_pickle=False)
