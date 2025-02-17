###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
"""
Helper functions for the S_k problem
"""
from qokit.fur.qaoa_simulator_base import TermsType
import numpy as np
from itertools import combinations


def sk_obj(x: np.ndarray, J: np.ndarray) -> float:
    """Compute the value of objective function for SK model.
    Args:
        x (numpy.ndarray): binary string as numpy array.
        J (numpy.ndarray): Matrix specifying the coupling in the SK model.
    Returns:
        float: value of the objective function.
    """
    n = len(x)
    X = np.outer(2 * x - 1, 2 * x - 1)
    return np.sum(J * X) / np.sqrt(n)  # type: ignore


def get_sk_terms(J: np.ndarray) -> TermsType:
    """Get terms corresponding to cost function value

    .. math::

        S = 1/\sqrt(N) \sum_{(i,j)\\in G} J_ij * (s_i*s_j)/2

    Args:
        J (numpy.ndarray): Matrix specifying the coupling in the SK model.
    Returns:
        terms to be used in the simulation
    """
    N = J.shape[0]

    terms = [((2 * J[i, j]) / np.sqrt(N), (int(i), int(j))) for i, j in combinations(range(N), 2)]
    return terms
