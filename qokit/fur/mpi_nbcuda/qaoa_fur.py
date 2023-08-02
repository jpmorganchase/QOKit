###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from collections.abc import Sequence
import numpy as np

from ..nbcuda.diagonal import apply_diagonal
from .fur import furx_all  # , furxy_ring, furxy_complete


def apply_qaoa_furx(sv: np.ndarray, gammas: Sequence[float], betas: Sequence[float], hc_diag: np.ndarray, n_local_qubits: int, n_all_qubits: int, comm) -> None:
    """
    apply a QAOA with the X mixer defined by
    U(beta) = sum_{j} exp(-i*beta*X_j/2)
    where X_j is the Pauli-X operator applied on the jth qubit.
    This operation is in-place to sv.
    """
    for gamma, beta in zip(gammas, betas):
        apply_diagonal(sv, gamma, hc_diag)
        furx_all(sv, beta, n_local_qubits, n_all_qubits, comm)
