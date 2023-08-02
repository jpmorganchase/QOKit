###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import math
import numba.cuda
import numpy as np


########################################
# single-qubit X rotation
########################################
@numba.cuda.jit
def furx_kernel(x, wa, wb, q, mask1, mask2):
    """CUDA kernel for fast uniform X rotations"""
    n_states = len(x)
    n_groups = n_states // 2
    tid = numba.cuda.grid(1)

    if tid < n_groups:
        ia = (tid & mask1) | ((tid & mask2) << 1)
        ib = ia | (1 << q)
        x[ia], x[ib] = wa * x[ia] + wb * x[ib], wb * x[ia] + wa * x[ib]


def furx(x: np.ndarray, theta: float, q: int):
    """
    Apply in-place fast Rx gate exp(-1j * theta * X) on qubit q to statevector array x.
    theta: rotation angle
    """
    n_states = len(x)
    mask1 = (1 << q) - 1
    mask2 = mask1 ^ ((n_states - 1) >> 1)
    furx_kernel.forall(n_states)(x, math.cos(theta), -1j * math.sin(theta), q, mask1, mask2)


def furx_all(x: np.ndarray, theta: float, n_qubits: int):
    """
    Apply in-place fast uniform Rx gates exp(-1j * theta * X) to statevector array x.
    theta: rotation angle
    """
    for i in range(n_qubits):
        furx(x, theta, i)


########################################
# two-qubit XX+YY rotation
########################################
@numba.cuda.jit
def furxy_kernel(x, wa, wb, q1, q2, mask1, mask2, maskm):
    """CUDA kernel for fast uniform XX + YY rotations"""
    n_states = len(x)
    n_groups = n_states // 4
    tid = numba.cuda.grid(1)

    if tid < n_groups:
        i0 = (tid & mask1) | ((tid & maskm) << 1) | ((tid & mask2) << 2)
        ia = i0 | (1 << q1)
        ib = i0 | (1 << q2)
        x[ia], x[ib] = wa * x[ia] + wb * x[ib], wb * x[ia] + wa * x[ib]


def furxy(x: np.ndarray, theta: float, q1: int, q2: int):
    """
    Applies e^{-i theta (XX + YY)} on q1, q2 to statevector x
    Same as XXPlusYYGate in Qiskit
    https://qiskit.org/documentation/stubs/qiskit.circuit.library.XXPlusYYGate.html
    theta: rotation angle
    """
    if q1 > q2:
        q1, q2 = q2, q1

    n_states = len(x)

    mask1 = (1 << q1) - 1
    mask2 = (1 << (q2 - 1)) - 1
    maskm = mask1 ^ mask2
    mask2 ^= (n_states - 1) >> 2

    furxy_kernel.forall(n_states)(x, math.cos(theta), -1j * math.sin(theta), q1, q2, mask1, mask2, maskm)


def furxy_ring(x: np.ndarray, theta: float, n_qubits: int):
    """
    Applies e^{-i theta (XX + YY)} on all adjacent pairs of qubits (with wrap-around)
    """
    for i in range(2):
        for j in range(i, n_qubits - 1, 2):
            furxy(x, theta, j, j + 1)
    furxy(x, theta, 0, n_qubits - 1)


def furxy_complete(x: np.ndarray, theta: float, n_qubits: int):
    """
    Applies e^{-i theta (XX + YY)} on all pairs of qubits
    """
    for i in range(n_qubits - 1):
        for j in range(i + 1, n_qubits):
            furxy(x, theta, i, j)
