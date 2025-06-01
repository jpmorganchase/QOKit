###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import math
import numpy as np
from numba import njit, prange

from qokit.config import USE_NUMBA
if USE_NUMBA:
    from numba import njit
else:
    def njit(*args, **kwargs):
        return lambda f: f

########################################
# single-qubit X rotation
########################################
@njit(fastmath=True, cache=True)
def furx(x: np.ndarray, theta: float, q: int) -> np.ndarray:
    """
    Applies e^{-i theta X} on qubit indexed by q
    """
    n_states = len(x)
    n_groups = n_states // 2

    mask1 = (1 << q) - 1
    mask2 = mask1 ^ ((n_states - 1) >> 1)

    wa = math.cos(theta)
    wb = -1j * math.sin(theta)

    for i in range(n_groups):
        ia = (i & mask1) | ((i & mask2) << 1)
        ib = ia | (1 << q)
        a = x[ia]
        b = x[ib]
        x[ia] = wa * a + wb * b
        x[ib] = wb * a + wa * b

    return x


@njit(fastmath=True, cache=True)
def furx_all(x: np.ndarray, theta: float, n_qubits: int) -> np.ndarray:
    """
    Applies e^{-i theta X} on all qubits
    """
    for i in range(n_qubits):
        furx(x, theta, i)
    return x


########################################
# two-qubit XX+YY rotation
########################################
@njit(fastmath=True, cache=True)
def furxy(x: np.ndarray, theta: float, q1: int, q2: int) -> np.ndarray:
    """
    Applies e^{-i theta (XX + YY)} on q1, q2
    Same as XXPlusYYGate in Qiskit
    https://qiskit.org/documentation/stubs/qiskit.circuit.library.XXPlusYYGate.html
    """

    if q1 > q2:
        q1, q2 = q2, q1

    n_states = len(x)
    n_groups = n_states // 4

    mask1 = (1 << q1) - 1
    mask2 = (1 << (q2 - 1)) - 1
    maskm = mask1 ^ mask2
    mask2 ^= (n_states - 1) >> 2

    wa = math.cos(theta)
    wb = -1j * math.sin(theta)

    for i in range(n_groups):
        i0 = (i & mask1) | ((i & maskm) << 1) | ((i & mask2) << 2)
        ia = i0 | (1 << q1)
        ib = i0 | (1 << q2)
        a = x[ia]
        b = x[ib]
        x[ia] = wa * a + wb * b
        x[ib] = wb * a + wa * b

    return x


@njit(fastmath=True, cache=True, parallel=True)
def furxy_ring(x: np.ndarray, theta: float, n_qubits: int) -> np.ndarray:
    """
    Applies e^{-i theta (XX + YY)} on all adjacent pairs of qubits (with wrap-around)
    Parallelized with prange for speedup.
    """
    for i in prange(2):
        for j in range(i, n_qubits - 1, 2):
            furxy(x, theta, j, j + 1)
    furxy(x, theta, 0, n_qubits - 1)

    return x


@njit(fastmath=True, cache=True)
def furxy_complete(x: np.ndarray, theta: float, n_qubits: int) -> np.ndarray:
    """
    Applies e^{-i theta (XX + YY)} on all pairs of qubits
    """
    for i in range(n_qubits - 1):
        for j in range(i + 1, n_qubits):
            furxy(x, theta, i, j)

    return x
