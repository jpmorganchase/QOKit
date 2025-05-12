###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from ..lazy_import import MPI
from ..nbcuda.fur import furx_all as furx_local
from ..nbcuda.fur import furx
import math


def furx_all(x, theta: float, n_local_qubits: int, n_all_qubits: int, comm):
    assert n_local_qubits <= n_all_qubits
    assert n_all_qubits <= 2 * n_local_qubits, "n_all_qubits > 2*n_local_qubits is not yet implemented"

    furx_local(x, theta, n_local_qubits)

    if n_all_qubits > n_local_qubits:
        import cupy as cp

        comm.Alltoall(MPI.IN_PLACE, x)
        a, b = math.cos(theta), -math.sin(theta)
        q_offset = 2 * n_local_qubits - n_all_qubits
        k_qubit = n_all_qubits - n_local_qubits
        n_states = len(x)
        state_mask = (n_states - 1) >> 1
        cp_x = cp.asarray(x)
        furx(cp_x, a, b, k_qubit, q_offset, state_mask)
        comm.Alltoall(MPI.IN_PLACE, x)
