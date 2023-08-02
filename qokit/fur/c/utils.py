###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from __future__ import annotations
from dataclasses import dataclass
import numba
import numpy as np


@numba.njit(parallel=True)
def combine_complex(real: np.ndarray, imag: np.ndarray) -> np.ndarray:
    return real + 1j * imag


@numba.njit(parallel=True)
def norm_squared(real: np.ndarray, imag: np.ndarray) -> np.ndarray:
    return real**2 + imag**2


@dataclass
class ComplexArray:
    real: np.ndarray
    imag: np.ndarray

    def get_complex(self) -> np.ndarray:
        return combine_complex(self.real, self.imag)

    def get_norm_squared(self) -> np.ndarray:
        return norm_squared(self.real, self.imag)


def get_complex_array(sv: ComplexArray | np.ndarray) -> ComplexArray:
    """
    create a ComplexArray from a NumPy array or return the object as is
    if it's already a ComplexArray
    """
    if not isinstance(sv, ComplexArray):
        sv = ComplexArray(sv.real.astype("float"), sv.imag.astype("float"))
    return sv
