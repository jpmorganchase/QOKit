###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import math
import numba.cuda
import numpy as np


@numba.cuda.jit
def apply_diagonal_kernel(sv, gamma, diag):
    n = len(sv)
    tid = numba.cuda.grid(1)
    if tid < n:
        x = 0.5 * gamma * diag[tid]
        sv[tid] *= math.cos(x) - 1j * math.sin(x)


def apply_diagonal(sv, gamma, diag):
    apply_diagonal_kernel.forall(len(sv))(sv, gamma, diag)


@numba.cuda.jit
def apply_diagonal_from_terms_kernel(sv, gamma, terms_coef, terms_mask, offset):
    """Apply the phase-separation layer without a precomputed diagonal.

    Computes the Ising energy for each basis state on-the-fly from the
    problem's weighted terms (coeff, pos_mask pairs), then applies the
    corresponding phase rotation to the statevector.  This eliminates the
    O(2^n) memory allocation and CPU→GPU transfer required by the precomputed
    diagonal approach.

    For MAXCUT problems with edge-count << 2^n this is faster than the lookup
    path for n >= ~22 qubits, because global-memory bandwidth is the dominant
    cost of the lookup while the on-the-fly arithmetic fits in registers.
    """
    n = len(sv)
    tid = numba.cuda.grid(1)
    if tid < n:
        state = tid + offset
        energy = 0.0
        for i in range(len(terms_coef)):
            parity = numba.cuda.popc(state & terms_mask[i]) & 1
            if parity:
                energy -= terms_coef[i]
            else:
                energy += terms_coef[i]
        x = 0.5 * gamma * energy
        sv[tid] *= math.cos(x) - 1j * math.sin(x)


def apply_diagonal_from_terms(sv, gamma, terms_coef_device, terms_mask_device, offset: int = 0):
    """Apply diagonal phase rotation with on-the-fly energy computation.

    Parameters
    ----------
    sv:
        CUDA device array (complex) of length 2^n — the statevector.
    gamma:
        Phase-separation angle for this QAOA layer.
    terms_coef_device:
        CUDA device array of float32 coefficients, one per Ising term.
    terms_mask_device:
        CUDA device array of int64 bitmasks, one per Ising term.
        Bit k is set iff qubit k participates in the term.
    offset:
        Rank-local offset added to tid before extracting bit values.
        Used by MPI backends where each rank holds a contiguous slice of
        the statevector.
    """
    apply_diagonal_from_terms_kernel.forall(len(sv))(sv, gamma, terms_coef_device, terms_mask_device, offset)


def terms_to_device_arrays(terms):
    """Convert a TermsType list into GPU-resident coefficient and mask arrays.

    Parameters
    ----------
    terms:
        Sequence of (coeff, [qubit_indices]) pairs — same format as
        ``QAOAFastSimulatorBase.__init__``'s *terms* parameter.

    Returns
    -------
    terms_coef_device, terms_mask_device : numba DeviceNDArray pair
        Ready to pass to :func:`apply_diagonal_from_terms`.
    """
    coefs = []
    masks = []
    for coef, positions in terms:
        mask = 0
        for p in positions:
            mask |= 1 << p
        coefs.append(float(coef))
        masks.append(mask)
    coef_arr = np.array(coefs, dtype=np.float32)
    mask_arr = np.array(masks, dtype=np.int64)
    return numba.cuda.to_device(coef_arr), numba.cuda.to_device(mask_arr)
