###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import typing
import numpy as np


def check_arrays(*arrs: np.ndarray) -> int:
    """check that all arrays have the same length and return the length"""
    n = len(arrs[0])
    for arr in arrs[1:]:
        assert n == len(arr), f"Input arrays do not have the same size: {', '.join(str(len(arr)) for arr in arrs)}"
    return n


def check_num_qubits(n_qubits, n_states):
    """check that the number of qubits and the number of states match"""
    assert n_states == 2**n_qubits, "state vector length {} and number of qubits {} do not match".format(n_states, n_qubits)


def furx(
    sv_real: np.ndarray,
    sv_imag: np.ndarray,
    theta: float,
    q: int,
) -> None:
    from .lib import _furx

    n_states = check_arrays(sv_real, sv_imag)
    _furx(
        sv_real,
        sv_imag,
        theta,
        q,
        n_states,
    )


def apply_qaoa_furx(
    sv_real: np.ndarray,
    sv_imag: np.ndarray,
    gammas: typing.Sequence[float],
    betas: typing.Sequence[float],
    hc_diag: np.ndarray,
    n_qubits: int,
) -> None:
    from .lib import _apply_qaoa_furx

    n_states = check_arrays(sv_real, sv_imag, hc_diag)
    n_layers = check_arrays(gammas, betas)
    check_num_qubits(n_qubits, n_states)
    _apply_qaoa_furx(
        sv_real,
        sv_imag,
        np.asarray(gammas, dtype="float"),
        np.asarray(betas, dtype="float"),
        hc_diag,
        n_qubits,
        n_states,
        n_layers,
    )


def furxy(
    sv_real: np.ndarray,
    sv_imag: np.ndarray,
    theta: float,
    q1: int,
    q2: int,
) -> None:
    from .lib import _furxy

    n_states = check_arrays(sv_real, sv_imag)
    _furxy(
        sv_real,
        sv_imag,
        theta,
        q1,
        q2,
        n_states,
    )


def apply_qaoa_furxy_ring(
    sv_real: np.ndarray,
    sv_imag: np.ndarray,
    gammas: typing.Sequence[float],
    betas: typing.Sequence[float],
    hc_diag: np.ndarray,
    n_qubits: int,
    n_trotters: int,
) -> None:
    from .lib import _apply_qaoa_furxy_ring

    n_states = check_arrays(sv_real, sv_imag, hc_diag)
    n_layers = check_arrays(gammas, betas)
    check_num_qubits(n_qubits, n_states)
    _apply_qaoa_furxy_ring(
        sv_real,
        sv_imag,
        np.asarray(gammas, dtype="float"),
        np.asarray(betas, dtype="float"),
        hc_diag,
        n_qubits,
        n_states,
        n_layers,
        n_trotters,
    )


def apply_qaoa_furxy_complete(
    sv_real: np.ndarray,
    sv_imag: np.ndarray,
    gammas: typing.Sequence[float],
    betas: typing.Sequence[float],
    hc_diag: np.ndarray,
    n_qubits: int,
    n_trotters: int,
) -> None:
    from .lib import _apply_qaoa_furxy_complete

    n_states = check_arrays(sv_real, sv_imag, hc_diag)
    n_layers = check_arrays(gammas, betas)
    check_num_qubits(n_qubits, n_states)
    _apply_qaoa_furxy_complete(
        sv_real,
        sv_imag,
        np.asarray(gammas, dtype="float"),
        np.asarray(betas, dtype="float"),
        hc_diag,
        n_qubits,
        n_states,
        n_layers,
        n_trotters,
    )
