###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from collections.abc import Sequence
import numpy as np
from .fur import furx_all, furxy_ring, furxy_complete


def apply_qaoa_furx(sv: np.ndarray, gammas: Sequence[float], betas: Sequence[float], hc_diag: np.ndarray, n_qubits: int) -> None:
    """
    apply a QAOA with the X mixer defined by
    U(beta) = sum_{j} exp(-i*beta*X_j/2)
    where X_j is the Pauli-X operator applied on the jth qubit.
    @param sv array NumPy array (dtype=complex) of length n containing the statevector
    @param gammas parameters for the phase separating layers
    @param betas parameters for the mixing layers
    @param hc_diag array of length n containing diagonal elements of the diagonal cost Hamiltonian
    @param n_qubits total number of qubits represented by the statevector
    """
    for gamma, beta in zip(gammas, betas):
        sv *= np.exp(-0.5j * gamma * hc_diag)
        furx_all(sv, beta, n_qubits)


def apply_qaoa_furxy_ring(sv: np.ndarray, gammas: Sequence[float], betas: Sequence[float], hc_diag: np.ndarray, n_qubits: int, n_trotters: int = 1) -> None:
    """
    apply a QAOA with the XY-ring mixer defined by
        U(beta) = sum_{j} exp(-i*beta*(X_{j}X_{j+1}+Y_{j}Y_{j+1})/4)
    where X_j and Y_j are the Pauli-X and Pauli-Y operators applied on the jth qubit, respectively.
    This operation is in-place to sv.
    @param sv array CUDA device array (dtype=complex) of length n containing the statevector
    @param gammas parameters for the phase separating layers
    @param betas parameters for the mixing layers
    @param hc_diag array of length n containing diagonal elements of the diagonal cost Hamiltonian
    @param n_qubits total number of qubits represented by the statevector
    @param n_trotters number of Trotter steps in each XY mixer layer
    """
    for gamma, beta in zip(gammas, betas):
        sv *= np.exp(-0.5j * gamma * hc_diag)
        for _ in range(n_trotters):
            furxy_ring(sv, beta / n_trotters, n_qubits)


def apply_qaoa_furxy_complete(sv: np.ndarray, gammas: Sequence[float], betas: Sequence[float], hc_diag: np.ndarray, n_qubits: int, n_trotters: int = 1) -> None:
    """
    apply a QAOA with the XY-complete mixer defined by
        U(beta) = sum_{j,k} exp(-i*beta*(X_{j}X_{k}+Y_{j}Y_{k})/4)
    where X_j and Y_j are the Pauli-X and Pauli-Y operators applied on the jth qubit, respectively.
    This operation is in-place to sv.
    @param sv array CUDA device array (dtype=complex) of length n containing the statevector
    @param gammas parameters for the phase separating layers
    @param betas parameters for the mixing layers
    @param hc_diag array of length n containing diagonal elements of the diagonal cost Hamiltonian
    @param n_qubits total number of qubits represented by the statevector
    @param n_trotters number of Trotter steps in each XY mixer layer
    """
    for gamma, beta in zip(gammas, betas):
        sv *= np.exp(-0.5j * gamma * hc_diag)
        for _ in range(n_trotters):
            furxy_complete(sv, beta / n_trotters, n_qubits)
