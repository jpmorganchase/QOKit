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


def furx_qudit(
    sv_real: np.ndarray,
    sv_imag: np.ndarray,
    theta: float,
    q: int,
    n_precision: int,
    A_mat: np.ndarray,
) -> None:
    from .lib import _furx_qudit

    # Ensure A_mat is complex and split into real and imaginary parts
    assert np.iscomplexobj(A_mat), "A_mat must be a complex matrix"
    A_mat_real = np.asarray(A_mat.real, dtype="float")
    A_mat_imag = np.asarray(A_mat.imag, dtype="float")

    # Validate input arrays
    n_states = check_arrays(sv_real, sv_imag)

    # Call the C function
    _furx_qudit(
        sv_real,
        sv_imag,
        theta,
        q,
        n_states,
        n_precision,
        A_mat_real,
        A_mat_imag,
    )

def apply_qaoa_furx_qudit(
    sv_real: np.ndarray,
    sv_imag: np.ndarray,
    gammas: typing.Sequence[float],
    betas: typing.Sequence[float],
    hc_diag: np.ndarray,
    A_mat: np.ndarray,
    n_precision:int,
    n_qubits:int
    

) -> None:
    from .lib import _apply_qaoa_furx_qudit

    # Ensure A_mat is complex and split into real and imaginary parts
    assert np.iscomplexobj(A_mat), "A_mat must be a complex matrix"
    A_mat_real = np.ascontiguousarray(A_mat.real, dtype=np.float64)
    A_mat_imag = np.ascontiguousarray(A_mat.imag, dtype=np.float64)
    # Validate input arrays
    n_states = check_arrays(sv_real, sv_imag, hc_diag)
    n_layers = check_arrays(gammas, betas)
    check_num_qubits(n_qubits, n_states)

    # Call the C function
    _apply_qaoa_furx_qudit(
        sv_real,
        sv_imag,
        np.asarray(gammas, dtype="float"),
        np.asarray(betas, dtype="float"),
        hc_diag,
        A_mat_real,
        A_mat_imag,
        n_precision,
        n_qubits,
        n_states,
        n_layers
        

    )    

