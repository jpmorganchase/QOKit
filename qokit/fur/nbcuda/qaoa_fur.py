###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from collections.abc import Sequence
from typing import Callable, Optional
import numpy as np

from .diagonal import apply_diagonal
from .fur import furx_all, furxy_ring, furxy_complete


def apply_qaoa_furx(
    sv: np.ndarray,
    gammas: Sequence[float],
    betas: Sequence[float],
    hc_diag: np.ndarray,
    n_qubits: int,
    *,
    apply_diag_fn: Optional[Callable] = None,
) -> None:
    """Apply a QAOA circuit with the X mixer.

    U(beta) = sum_{j} exp(-i*beta*X_j/2)

    Parameters
    ----------
    sv:
        CUDA device array (complex) of length 2^n — modified in-place.
    gammas:
        Phase-separation angles, one per QAOA layer.
    betas:
        Mixing angles, one per QAOA layer.
    hc_diag:
        Precomputed diagonal of the cost Hamiltonian (device array).
        Ignored when *apply_diag_fn* is provided.
    n_qubits:
        Total qubit count.
    apply_diag_fn:
        Optional ``(sv, gamma) -> None`` callable that applies the
        phase-separation layer.  Pass the simulator's
        ``_apply_diagonal_phase`` to enable lazy on-the-fly energy
        computation (Issue #35).
    """
    diag_apply = apply_diag_fn if apply_diag_fn is not None else (lambda _sv, _g: apply_diagonal(_sv, _g, hc_diag))
    for gamma, beta in zip(gammas, betas):
        diag_apply(sv, gamma)
        furx_all(sv, beta, n_qubits)


def apply_qaoa_furxy_ring(
    sv: np.ndarray,
    gammas: Sequence[float],
    betas: Sequence[float],
    hc_diag: np.ndarray,
    n_qubits: int,
    n_trotters: int = 1,
    *,
    apply_diag_fn: Optional[Callable] = None,
) -> None:
    """Apply a QAOA circuit with the XY-ring mixer.

    U(beta) = sum_{j} exp(-i*beta*(X_{j}X_{j+1}+Y_{j}Y_{j+1})/4)

    Parameters match ``apply_qaoa_furx``; see its docstring for
    *apply_diag_fn* semantics.
    """
    diag_apply = apply_diag_fn if apply_diag_fn is not None else (lambda _sv, _g: apply_diagonal(_sv, _g, hc_diag))
    for gamma, beta in zip(gammas, betas):
        diag_apply(sv, gamma)
        for _ in range(n_trotters):
            furxy_ring(sv, beta / n_trotters, n_qubits)


def apply_qaoa_furxy_complete(
    sv: np.ndarray,
    gammas: Sequence[float],
    betas: Sequence[float],
    hc_diag: np.ndarray,
    n_qubits: int,
    n_trotters: int = 1,
    *,
    apply_diag_fn: Optional[Callable] = None,
) -> None:
    """Apply a QAOA circuit with the XY-complete mixer.

    U(beta) = sum_{j,k} exp(-i*beta*(X_{j}X_{k}+Y_{j}Y_{k})/4)

    Parameters match ``apply_qaoa_furx``; see its docstring for
    *apply_diag_fn* semantics.
    """
    diag_apply = apply_diag_fn if apply_diag_fn is not None else (lambda _sv, _g: apply_diagonal(_sv, _g, hc_diag))
    for gamma, beta in zip(gammas, betas):
        diag_apply(sv, gamma)
        for _ in range(n_trotters):
            furxy_complete(sv, beta / n_trotters, n_qubits)
