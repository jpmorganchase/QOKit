###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from __future__ import annotations
import numpy as np
from .utils import ComplexArray, get_complex_array
from . import csim


def furx(sv: ComplexArray | np.ndarray, theta: float, q: int) -> ComplexArray:
    """
    apply to a statevector a single-qubit Pauli-X rotation defined by
    Rx(theta) = exp(-i*theta*X/2)
    where X is the Pauli-X operator.
    The operation will be in-place if the input is a ComplexArray, otherwise (NumPy array)
    the input will be copied to a new ComplexArray.
    @param sv statevector on which the rotation is applied (ComplexArray or numpy.ndarray)
    @param theta rotation angle
    @param q index of qubit on which the rotation is applied
    @return a ComplexArray containing the statevector after the operation
    """
    sv = get_complex_array(sv)
    csim.furx(sv.real, sv.imag, 0.5 * theta, q)
    return sv


def furxy(sv: ComplexArray | np.ndarray, theta: float, q1: int, q2: int) -> ComplexArray:
    """
    apply to a statevector a two-qubit XX+YY rotation defined by
    Rxy(theta) = exp(-i*theta*(XX+YY)/4)
    where X and Y are the Pauli-X and Pauli-Y operators, respectively.
    The operation will be in-place if the input is a ComplexArray, otherwise (NumPy array)
    the input will be copied to a new ComplexArray.
    @param sv statevector on which the rotation is applied (ComplexArray or numpy.ndarray)
    @param theta rotation angle
    @param q1 index of the first qubit on which the rotation is applied
    @param q2 index of the second qubit on which the rotation is applied
    @return a ComplexArray containing the statevector after the operation
    """
    sv = get_complex_array(sv)
    csim.furxy(sv.real, sv.imag, 0.5 * theta, q1, q2)
    return sv
