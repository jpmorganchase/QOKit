###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import numpy as np
from sympy import kronecker_product
from qokit.matrix import kron


def test_kron():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    expected_result = np.array([[5, 6, 10, 12], [7, 8, 14, 16], [15, 18, 20, 24], [21, 24, 28, 32]])
    assert np.array_equal(kron(A, B), expected_result)
