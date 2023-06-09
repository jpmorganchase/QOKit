###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
"""
Script used to precompute the energies in ../qokit/assets/precomputed_energies/
"""

import sys

sys.path.append("../code/")

import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("N", help="number of spins", type=int)
    parser.add_argument("nchunks", help="number of chunks", type=int)
    parser.add_argument("postfix_width", help="width of the postfix", type=int)
    args = parser.parse_args()

    N = args.N
    precomputed_energies = np.array([])

    for postfix_id in tqdm(range(args.nchunks)):
        postfix = [int(x) for x in f"{postfix_id:0{args.postfix_width}b}"][::-1]
        chunkpath = f"precomputed_energies_workspace/precomputed_energies_{N}_{postfix_id}.npy"
        if not Path(chunkpath).exists():
            raise ValueError(f"Chunk missing in {chunkpath}")
        precomputed_energies = np.hstack([precomputed_energies, np.load(chunkpath)])

    outpath = f"../qokit/assets/precomputed_energies/precomputed_energies_{N}.npy"
    np.save(outpath, precomputed_energies, allow_pickle=False)
