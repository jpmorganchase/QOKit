###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from copy import deepcopy
import time
import numpy as np
import pickle
from pathlib import Path

import sys

sys.path.append("../code/")

from labs import true_optimal_mf
from qaoa_objective_labs import (
    get_precomputed_labs_merit_factors,
    get_precomputed_optimal_bitstrings,
)

for N in range(20, 34):
    start = time.time()
    outpath = f"data/precomputed_top_energy_levels_{N}.pickle"
    if Path(outpath).exists():
        print(f"Found precomputed pickle at {outpath}, skipping", flush=True)
        continue

    energies = (-(N**2) / (2 * get_precomputed_labs_merit_factors(N))).astype(int)
    unique_energies = sorted(set(energies))
    unique_merit_factors = N**2 / (2 * np.array(unique_energies))
    print(f"{true_optimal_mf[N]} == {unique_merit_factors[0]:.3f}")

    num_bitstrings_per_energy_level = {idx: sum(energies == en) for idx, en in enumerate(unique_energies[:10])}

    assert num_bitstrings_per_energy_level[0] == len(get_precomputed_optimal_bitstrings(N))

    row = {
        "N": N,
        "unique_energies": deepcopy(unique_energies),
        "unique_merit_factors": deepcopy(unique_merit_factors),
        "num_bitstrings_per_energy_level": deepcopy(num_bitstrings_per_energy_level),
    }
    pickle.dump(row, open(outpath, "wb"))
    end = time.time()

    print(f"Done for N={N} in {end-start:.2f}")
    print(true_optimal_mf[N], len(unique_merit_factors), unique_merit_factors[:10])
