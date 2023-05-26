###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import ctypes
import typing
from pathlib import Path
import numpy as np
from numpy.ctypeslib import ndpointer


code_dir = Path(__file__).parent
try:
    lib = ctypes.cdll.LoadLibrary(code_dir / "libcsim.so")
except OSError as e:
    raise OSError("You must compile the C simulator before running the code. Please follow the instructions in README.md") from e


def check_arrays(*arrs: np.ndarray) -> int:
    """check that all arrays have the same length and return the length"""
    n = len(arrs[0])
    for arr in arrs[1:]:
        assert n == len(arr), f"Input arrays do not have the same size: {', '.join(str(len(arr)) for arr in arrs)}"
    return n


def check_num_qubits(n_qubits, n_states):
    """check that the number of qubits and the number of states match"""
    assert n_states == 2**n_qubits, "state vector length {} and number of qubits {} do not match".format(n_states, n_qubits)


_furx = lib.furx
_furx.restype = None
_furx.argtypes = [
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_double,
    ctypes.c_uint,
    ctypes.c_size_t,
]


def furx(
    sv_real: np.ndarray,
    sv_imag: np.ndarray,
    theta: float,
    q: int,
) -> None:
    n_states = check_arrays(sv_real, sv_imag)
    _furx(
        sv_real,
        sv_imag,
        theta,
        q,
        n_states,
    )


_apply_qaoa_furx = lib.apply_qaoa_furx
_apply_qaoa_furx.restype = None
_apply_qaoa_furx.argtypes = [
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_uint,
    ctypes.c_size_t,
    ctypes.c_size_t,
]


def apply_qaoa_furx(
    sv_real: np.ndarray,
    sv_imag: np.ndarray,
    gammas: typing.Sequence[float],
    betas: typing.Sequence[float],
    hc_diag: np.ndarray,
    n_qubits: int,
) -> None:
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


_furxy = lib.furxy
_furxy.restype = None
_furxy.argtypes = [
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_double,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_size_t,
]


def furxy(
    sv_real: np.ndarray,
    sv_imag: np.ndarray,
    theta: float,
    q1: int,
    q2: int,
) -> None:
    n_states = check_arrays(sv_real, sv_imag)
    _furxy(
        sv_real,
        sv_imag,
        theta,
        q1,
        q2,
        n_states,
    )


_apply_qaoa_furxy_ring = lib.apply_qaoa_furxy_ring
_apply_qaoa_furxy_ring.restype = None
_apply_qaoa_furxy_ring.argtypes = [
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_uint,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
]


def apply_qaoa_furxy_ring(
    sv_real: np.ndarray,
    sv_imag: np.ndarray,
    gammas: typing.Sequence[float],
    betas: typing.Sequence[float],
    hc_diag: np.ndarray,
    n_qubits: int,
    n_trotters: int,
) -> None:
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


_apply_qaoa_furxy_complete = lib.apply_qaoa_furxy_complete
_apply_qaoa_furxy_complete.restype = None
_apply_qaoa_furxy_complete.argtypes = [
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_uint,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
]


def apply_qaoa_furxy_complete(
    sv_real: np.ndarray,
    sv_imag: np.ndarray,
    gammas: typing.Sequence[float],
    betas: typing.Sequence[float],
    hc_diag: np.ndarray,
    n_qubits: int,
    n_trotters: int,
) -> None:
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
