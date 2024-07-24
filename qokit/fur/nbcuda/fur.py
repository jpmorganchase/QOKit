###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import math
import numba.cuda
import numpy as np
import cupy as cp
from pathlib import Path
from functools import lru_cache


########################################
# single-qubit X rotation
########################################
@lru_cache
def get_furx_kernel(k_qubits: int, q_offset: int, state_mask: int):
    """
    Generate furx kernel for the specified sub-group size.
    """
    if k_qubits > 6:
        kernel_name = f"furx_kernel<{k_qubits},{q_offset},{state_mask}>"
    else:
        kernel_name = f"warp_furx_kernel<{k_qubits},{q_offset}>"

    code = open(Path(__file__).parent / "furx.cu").read()
    return cp.RawModule(code=code, name_expressions=[kernel_name], options=("-std=c++17",)).get_function(kernel_name)


def furx(sv: cp.ndarray, a: float, b: float, k_qubits: int, q_offset: int, state_mask: int):
    """
    Apply in-place fast Rx gate exp(-1j * theta * X) on k consequtive qubits to statevector array x.

    sv: statevector
    a: cosine factor
    b: sine factor
    k_qubits: number of qubits to process concurrently
    q_offset: starting qubit number
    state_mask: mask for indexing
    """
    if k_qubits > 11:
        raise ValueError("k_qubits should be <= 11 because of shared memory constraints")

    seq_kernel = get_furx_kernel(k_qubits, q_offset, state_mask)

    if k_qubits > 6:
        threads = 1 << (k_qubits - 1)
    else:
        threads = min(32, len(sv))

    seq_kernel(((len(sv) // 2 + threads - 1) // threads,), (threads,), (sv, a, b))


def furx_all(sv: np.ndarray, theta: float, n_qubits: int):
    """
    Apply in-place fast uniform Rx gates exp(-1j * theta * X) to statevector array x.

    sv: statevector
    theta: rotation angle
    n_qubits: total number of qubits
    """
    n_states = len(sv)
    state_mask = (n_states - 1) >> 1

    a, b = math.cos(theta), -math.sin(theta)

    group_size = 10
    last_group_size = n_qubits % group_size

    cp_sv = cp.asarray(sv)

    for q_offset in range(0, n_qubits - last_group_size, group_size):
        furx(cp_sv, a, b, group_size, q_offset, state_mask)

    if last_group_size > 0:
        furx(cp_sv, a, b, last_group_size, n_qubits - last_group_size, state_mask)


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
