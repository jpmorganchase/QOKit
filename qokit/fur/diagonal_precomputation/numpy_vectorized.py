###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import numpy as np


def bit_count(arr):
    # Make the values type-agnostic (as long as it's integers)
    t = arr.dtype.type
    mask = t(-1)
    s55 = t(0x5555555555555555 & mask)  # Add more digits for 128bit support
    s33 = t(0x3333333333333333 & mask)
    s0F = t(0x0F0F0F0F0F0F0F0F & mask)
    s01 = t(0x0101010101010101 & mask)

    arr = arr - ((arr >> 1) & s55)
    arr = (arr & s33) + ((arr >> 2) & s33)
    arr = (arr + (arr >> 4)) & s0F
    return (arr * s01) >> (8 * (arr.itemsize - 1))


def precompute_vectorized_cpu_parallel(weighted_terms, offset, N):
    state_indices = np.arange(2**N)
    term_v = np.zeros_like(state_indices, dtype=np.float64)
    for coeff, pos in weighted_terms:
        a = sum([2**x for x in pos])
        tmp = np.bitwise_and(a, state_indices)
        term_v += coeff * (1 - 2 * (bit_count(tmp) % 2))
    term_v += offset
    return term_v
