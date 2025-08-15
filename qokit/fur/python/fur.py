###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import math
import numpy as np
from scipy.linalg import logm,expm
from itertools import product
import itertools
from collections.abc import Sequence
import numpy.typing as npt
 
def apply_on_qubits(
    gate: npt.NDArray[np.complex128], qubits: Sequence[int], state: npt.NDArray[np.complex128]
) -> npt.NDArray[np.complex128]:
    """Apply the given gate on the given qubits of the given state."""
    num_state_qubits = int(np.log2(len(state)))
    state_tensor = state.reshape([2] * num_state_qubits)

    num_gate_qubits = len(qubits)
    assert gate.shape == (2**num_gate_qubits,) * 2
    gate_tensor = gate.reshape([2] * num_gate_qubits * 2)

    # contract the gate with the state
    gate_axes = list(range(num_gate_qubits, 2 * num_gate_qubits))
    state_axes = list(qubits)
    state_tensor = np.tensordot(gate_tensor, state_tensor, axes=[gate_axes, state_axes])

    # move qubits to their locations, flatten the tensor, and return
    return np.moveaxis(state_tensor, range(num_gate_qubits), qubits).ravel()

# def furx_qudit(x: np.ndarray, 
#                theta: float,
#                A_mat: np.ndarray, 
#                target_qubits: list[int],
#                n_qubits: int) -> np.ndarray:
#     """
#     Apply a qudit operator (k-qubit unitary) U=expm(-i\theta A), 
#     where A is the generator of the QFT matrix. The target qubits
#     define the qudit and n_qubits is the total number of qubits which 
#     decomposes the entire system.
#     """
#     U = expm(-1.0j * theta * A_mat)
#     assert U.shape == (2**len(target_qubits), 2**len(target_qubits))

#     target_qubits = list(target_qubits)
#     k = len(target_qubits)
#     all_qubit_indices = list(range(n_qubits))
#     control_qubits = [q for q in all_qubit_indices if q not in target_qubits]
#     num_control = len(control_qubits)

#     for control_bits in product([0, 1], repeat=num_control):
#         base_index = 0
#         for q, b in zip(control_qubits, control_bits):
#             base_index |= (b << q)
#         indices = []
#         for target_bits in product([0, 1], repeat=k):
#             idx = base_index
#             for tq, tb in zip(target_qubits, target_bits):
#                 if tb:
#                     idx |= (1 << tq)
#                 else:
#                     idx &= ~(1 << tq)
#             indices.append(idx)
#         vec = x[indices]
#         x[indices] = U @ vec
#     return x


def furx_qudit(x: np.ndarray, 
               theta: float,
               A_mat:np.ndarray, 
               target_qubits:list[int],
               n_qubits:int) -> np.ndarray:
 
    
    """
    Apply a qudit operator (k-qubit unitary) U=expm(-i\theta A), 
    where A is the generator of the QFT matrix. The target qubits
    define the qudit and n_qubits is the total number of qubits which 
    decomposes the entire system.
    """
    
    U=expm(-1.0j*theta*A_mat)
    
    assert U.shape == (2**len(target_qubits), 2**len(target_qubits))
    #x = x.copy()
    target_qubits = list(target_qubits)
    k = len(target_qubits)
    # All possible configurations of the remaining (non-target) qubits
    all_qubit_indices = list(range(n_qubits))
    control_qubits = [q for q in all_qubit_indices if q not in target_qubits]
    num_control = len(control_qubits)
    for control_bits in product([0, 1], repeat=num_control):
        # build base index for fixed control bits
        base_index = 0
        for q, b in zip(control_qubits, control_bits):
            base_index |= (b << q)
        # build all 2^k indices where target bits vary
        indices = []
        for target_bits in product([0, 1], repeat=k):
            idx = base_index
            for tq, tb in zip(target_qubits, target_bits):
                if tb:
                    idx |= (1 << tq)
                else:
                    idx &= ~(1 << tq)
            indices.append(idx)
        # Extract the amplitudes, apply U, and update
        vec = x[indices]
        x[indices] = U @ vec
    return x



def furx_all_qudit(x: np.ndarray, theta: float, n_qubits: int,n_precision:int,A_mat:np.ndarray) -> np.ndarray:
    
    """
    Applies e^{-i theta X} on all qubits with different rotation for each qubit inside a qudit
    
    Parameters:
    ----------
    x:np.darray
      is the Pauli X
    theta: float
      the angle of rotation
    n_qubits: int
       the total number of qubits
    n_precision: int
       the number of bits of precision
       
    Return:
    ------
    x: np.ndarray
       the gate after rotation 
    """
    
    num_qudits=n_qubits//n_precision
    #print(num_qudits)
    
    for i in range(num_qudits):
        target_qubits=list(range(i*(n_precision),(i+1)*n_precision))
        furx_qudit(x, theta, A_mat, target_qubits,n_qubits)
    return x

# def furx_all_qudit(x: np.ndarray, theta: float, n_qubits: int, n_precision: int, A_mat: np.ndarray, strategy: str = 'tensor') -> np.ndarray:
#     """
#     Applies e^{-i theta X} on all qubits with different rotation for each qubit inside a qudit
#     """
#     num_qudits = n_qubits // n_precision
#     U = expm(-1.0j * theta * A_mat)

#     for i in range(num_qudits):
#         target_qubits = list(range(i * n_precision, (i + 1) * n_precision))
#         if strategy == 'tensor':
#             x = apply_on_qubits(U, target_qubits, x)
#         elif strategy == 'qudit':
#             x = furx_qudit(x, theta, A_mat, target_qubits, n_qubits)
#         else :
#             x = apply_unitary_on_subset(x, U, target_qubits, n_qubits)
#     return x



# def furx_all_qudit(x: np.ndarray, theta: float, n_qubits: int,n_precision:int,A_mat:np.ndarray) -> np.ndarray:
    
#     """
#     Applies e^{-i theta X} on all qubits with different rotation for each qubit inside a qudit
    
#     Parameters:
#     ----------
#     x:np.darray
#       is the Pauli X
#     theta: float
#       the angle of rotation
#     n_qubits: int
#        the total number of qubits
#     n_precision: int
#        the number of bits of precision
       
#     Return:
#     ------
#     x: np.ndarray
#        the gate after rotation 
#     """
    
#     num_qudits=n_qubits//n_precision
#     #print(num_qudits)
    
#     U=expm(-1.0j*theta*A_mat)
#     x_copy1=x.copy()
#     x_copy2=x.copy()
#     for i in range(num_qudits):
#         target_qubits=list(range(i*(n_precision),(i+1)*n_precision))
#         furx_qudit(x_copy1, theta, A_mat, target_qubits,n_qubits)
#         #print(target_qubits)
#         x_copy2=apply_on_qubits(U, target_qubits,x_copy2)
#     assert np.allclose(x_copy1,x_copy2)
#         #print(x)
#     return x_copy1


########################################
# single-qubit X rotation
########################################
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
        x[ia], x[ib] = wa * x[ia] + wb * x[ib], wb * x[ia] + wa * x[ib]

    return x


def furx_all(x: np.ndarray, theta: float, n_qubits: int,n_precision:int,is_precision:bool) -> np.ndarray:
    
    """
    Applies e^{-i theta X} on all qubits with different rotation for each qubit inside a qudit
    
    Parameters:
    ----------
    x:np.darray
      is the Pauli X
    theta: float
      the angle of rotation
    n_qubits: int
       the total number of qubits
    n_precision: int
       the number of bits of precision
       
    Return:
    ------
    x: np.ndarray
       the gate after rotation 
    """
    
    for i in range(n_qubits):
        # if is_precision:
        #     # val=np.random.uniform(0.95,1)
        #     val=1/(2**i)
        # else:
        #     val=1
        furx(x, theta, i)
    return x


########################################
# two-qubit XX+YY rotation
########################################
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
        x[ia], x[ib] = wa * x[ia] + wb * x[ib], wb * x[ia] + wa * x[ib]

    return x


def furxy_ring(x: np.ndarray, theta: float, n_qubits: int) -> np.ndarray:
    """
    Applies e^{-i theta (XX + YY)} on all adjacent pairs of qubits (with wrap-around)
    """
    for i in range(2):
        for j in range(i, n_qubits - 1, 2):
            furxy(x, theta, j, j + 1)
    furxy(x, theta, 0, n_qubits - 1)

    return x


def furxy_complete(x: np.ndarray, theta: float, n_qubits: int) -> np.ndarray:
    """
    Applies e^{-i theta (XX + YY)} on all pairs of qubits
    """
    for i in range(n_qubits - 1):
        for j in range(i + 1, n_qubits):
            furxy(x, theta, i, j)

    return x
