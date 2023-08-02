###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from ..lazy_import import MPI
from ..nbcuda.fur import furx as furx_local


def furx_all(x, theta: float, n_local_qubits: int, n_all_qubits: int, comm):
    assert n_local_qubits <= n_all_qubits
    assert n_all_qubits <= 2 * n_local_qubits, "n_all_qubits > 2*n_local_qubits is not yet implemented"

    for i in range(n_local_qubits):
        furx_local(x, theta, i)

    if n_all_qubits > n_local_qubits:
        comm.Alltoall(MPI.IN_PLACE, x)
        for i in range(2 * n_local_qubits - n_all_qubits, n_local_qubits):
            furx_local(x, theta, i)
        comm.Alltoall(MPI.IN_PLACE, x)
