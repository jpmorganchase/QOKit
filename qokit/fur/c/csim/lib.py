###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import ctypes
from numpy.ctypeslib import ndpointer

from .libpath import libpath


try:
    lib = ctypes.cdll.LoadLibrary(libpath)
except OSError as e:
    raise ImportError("You must compile the C simulator before running the code. Please follow the instructions in README.md") from e


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

_apply_qaoa_furx_qudit = lib.apply_qaoa_furx_qudit
_apply_qaoa_furx_qudit.restype = None
_apply_qaoa_furx_qudit.argtypes = [
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # sv_real
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # sv_imag
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # gammas
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # betas
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # hc_diag
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # A_mat_real
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # A_mat_imag
    ctypes.c_uint,                                    # n_precision
    ctypes.c_uint,                                    # n_qubits
    ctypes.c_size_t,                                  # n_states
    ctypes.c_size_t,                                  # n_layers
]



