###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import numpy as np
from .utils import get_complex_array
from . import fur


def furx(sv: np.ndarray, theta: float, q: int) -> np.ndarray:
    """
    a single-qubit Pauli-X rotation defined by
    Rx(theta) = exp(-i*theta*X/2)
    where X is the Pauli-X operator.
    The operation will be in-place if the input is a numpy.ndarray of complex
    data type. Otherwise the input will be copied to a new numpy.ndarray.
    @param sv statevector on which the rotation is applied
    @param theta rotation angle
    @param q qubit index to apply the rotation
    @return statevector after the rotation
    """
    sv = get_complex_array(sv)
    fur.furx(sv, 0.5 * theta, q)
    return sv


def furxy(sv: np.ndarray, theta: float, q1: int, q2: int) -> np.ndarray:
    """
    apply to a statevector a two-qubit XX+YY rotation defined by
    Rxy(theta) = exp(-i*theta*(XX+YY)/4)
    where X and Y are the Pauli-X and Pauli-Y operators, respectively.
    The operation will be in-place if the input is a numpy.ndarray of complex
    data type. Otherwise the input will be copied to a new numpy.ndarray.
    @param sv statevector on which the rotation is applied
    @param theta rotation angle
    @param q1 index of the first qubit on which the rotation is applied
    @param q2 index of the second qubit on which the rotation is applied
    @return statevector after the rotation
    """
    sv = get_complex_array(sv)
    fur.furxy(sv, 0.5 * theta, q1, q2)
    return sv
