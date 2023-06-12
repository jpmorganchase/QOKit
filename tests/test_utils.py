###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import numpy as np
from qokit.labs import get_energy_term_indices, negative_merit_factor_from_bitstring
from qokit.utils import precompute_energies_parallel, precompute_energies


def test_precompute_energies():
    N = 5
    terms, offset = get_energy_term_indices(N)
    ens1 = precompute_energies(negative_merit_factor_from_bitstring, N, N)
    ens2 = precompute_energies_parallel(negative_merit_factor_from_bitstring, N, 2)

    assert np.allclose(ens1, ens2)
