###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import numpy as np
import json
from pathlib import Path

import sys

sys.path.append("../qokit/")

from qaoa_objective_labs import get_precomputed_optimal_bitstrings

all_corrs = {}

for N in range(20, 31):
    outpath = f"data/correlation_between_cost_and_hamm_dist_{N}.npy"
    if Path(outpath).exists():
        print(f"Found precomputed pickle at {outpath}, skipping", flush=True)
        all_corrs[N] = np.load(outpath)[0, 1]
        continue
    print(f"N = {N}")

    arr = np.load(f"../qokit/assets/precomputed_merit_factors/precomputed_energies_{N}.npy")
    bit_strings = (((np.array(range(2**N))[:, None] & (1 << np.arange(N)))) > 0).astype(int)
    opt_bs = get_precomputed_optimal_bitstrings(N)
    all_hamm_dists = np.vstack([np.count_nonzero(bit_strings != opt_bs_one, axis=1) for opt_bs_one in opt_bs])
    hamm_dist = np.min(all_hamm_dists, axis=0)

    assert sum(hamm_dist == 0) == len(opt_bs)

    corr = np.corrcoef(arr, hamm_dist)
    print(corr)
    np.save(outpath, corr, allow_pickle=False)

json.dump(
    all_corrs,
    open("../qokit/assets/correlations_between_cost_and_hamm_dist.json", "w"),
    indent=4,
)
