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

_furx = lib.furx
_furx.restype = None
_furx.argtypes = [
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_double,
    ctypes.c_uint,
    ctypes.c_size_t,
]

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

_apply_qaoa_furx_int = lib.apply_qaoa_furx_int
_apply_qaoa_furx_int.restype = None
_apply_qaoa_furx_int.argtypes = [
    ndpointer(ctypes.c_int16, flags="C_CONTIGUOUS"),  # real_q
    ndpointer(ctypes.c_int16, flags="C_CONTIGUOUS"),  # imag_q
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # scale
    ctypes.c_uint,                                    # quant_bits
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), # gammas
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), # betas
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), # diag
    ctypes.c_uint,                                    # n_qubits
    ctypes.c_size_t,                                  # n_states
    ctypes.c_size_t                                   # p
]

# Export symbols for use in wrapper.py
__all__ = [
    "_furx",
    "_apply_qaoa_furx",
    "_apply_qaoa_furx_int",
    "_furxy",
    "_apply_qaoa_furxy_ring",
    "_apply_qaoa_furxy_complete"
]
