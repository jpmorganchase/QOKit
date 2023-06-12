###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import numpy as np

##############################
# Matrix utilities
##############################
X = np.array([[0, 1], [1, 0]])

Z = np.array([[1, 0], [0, -1]])

Y = -1j * Z @ X

I = np.identity(2)


def kron(*args: np.ndarray) -> np.ndarray:
    """
    Compute Kronecker product of matrices
    :param args: numpy.array
    :return: Kronecker product
    """
    if len(args) == 1:
        return args[0]
    return np.kron(args[0], kron(*args[1:]))
