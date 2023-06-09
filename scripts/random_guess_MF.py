###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import json

import sys

sys.path.append("../code/")

from qaoa_objective_labs import get_random_guess_merit_factor

all_random_MFs = {}

for N in range(10, 34):
    all_random_MFs[N] = get_random_guess_merit_factor(N)
    print(f"{N}\t{all_random_MFs[N]}")

json.dump(all_random_MFs, open("../qokit/assets/precomputed_random_guess_merit_factors.json", "w"))
