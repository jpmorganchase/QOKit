###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import numpy as np
from qokit.labs import get_energy_term_indices, negative_merit_factor_from_bitstring
from qokit.utils import precompute_energies_parallel, precompute_energies
from qokit.utils import reverse_array_index_bit_order


def test_precompute_energies():
    N = 5
    terms, offset = get_energy_term_indices(N)
    ens1 = precompute_energies(negative_merit_factor_from_bitstring, N, N)
    ens2 = precompute_energies_parallel(negative_merit_factor_from_bitstring, N, 2)

    assert np.allclose(ens1, ens2)


def test_reverse_array_index_bit_order():
    a = np.arange(16)
    b = reverse_array_index_bit_order(a)
    index = "1110"
    assert b[int(index, 2)] == a[int(index[::-1], 2)]
    index = "0001"
    assert b[int(index, 2)] == a[int(index[::-1], 2)]

    # the test should use an array larger than 2^8
    a = np.arange(1024)
    b = reverse_array_index_bit_order(a)
    index = "0000000001"
    assert b[int(index, 2)] == a[int(index[::-1], 2)]
    index = "1111001100"
    assert b[int(index, 2)] == a[int(index[::-1], 2)]
    assert a.sum() == b.sum()
