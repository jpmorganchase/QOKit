###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import typing
import numpy as np
from qokit.fur.c.quant_utils import quantise_fp, dequantise_fp

from .lib import (
    _furx,
    _apply_qaoa_furx,
    _apply_qaoa_furx_int,
    _furxy,
    _apply_qaoa_furxy_ring,
    _apply_qaoa_furxy_complete
)

def check_arrays(*arrs: np.ndarray) -> int:
    n = len(arrs[0])
    for arr in arrs[1:]:
        assert n == len(arr), f"Array size mismatch: {', '.join(str(len(arr)) for arr in arrs)}"
    return n

def check_num_qubits(n_qubits, n_states):
    assert n_states == 2**n_qubits, f"Statevector length {n_states} doesn't match 2^{n_qubits} qubits."

# ─────────────────────────────────────────────────────────────────────────────

def furx(sv_real, sv_imag, theta, q):
    n_states = check_arrays(sv_real, sv_imag)
    _furx(sv_real, sv_imag, theta, q, n_states)

# ─────────────────────────────────────────────────────────────────────────────

def apply_qaoa_furx(
    sv_real: np.ndarray,
    sv_imag: np.ndarray,
    gammas: typing.Sequence[float],
    betas: typing.Sequence[float],
    hc_diag: np.ndarray,
    n_qubits: int,
    quant_bits: int = 0,
    block_size: int = 1024,
):
    n_states = check_arrays(sv_real, sv_imag, hc_diag)
    n_layers = check_arrays(gammas, betas)
    check_num_qubits(n_qubits, n_states)

    if quant_bits > 0:
        sv_complex = sv_real + 1j * sv_imag
        rq, iq, scale = quantise_fp(sv_complex.astype(np.complex64), bits=quant_bits, block_size=block_size)

        # 🛡️ Ensure correct dtypes for C call
        rq = np.ascontiguousarray(rq.astype(np.int16 if quant_bits > 8 else np.int8))
        iq = np.ascontiguousarray(iq.astype(np.int16 if quant_bits > 8 else np.int8))
        scale = np.ascontiguousarray(scale.astype(np.float32))
        gammas = np.ascontiguousarray(gammas, dtype=np.float64)
        betas = np.ascontiguousarray(betas, dtype=np.float64)
        hc_diag = np.ascontiguousarray(hc_diag, dtype=np.float64)

        _apply_qaoa_furx_int(
            rq, iq, scale,
            quant_bits,
            gammas,
            betas,
            hc_diag,
            n_qubits,
            n_states,
            n_layers
        )

        # Only dequant once for both real and imag
        deq = dequantise_fp(rq, iq, scale, bits=quant_bits, block_size=block_size)
        sv_real[:] = deq.real.astype("float64")
        sv_imag[:] = deq.imag.astype("float64")
    else:
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

# ─────────────────────────────────────────────────────────────────────────────

def furxy(sv_real, sv_imag, theta, q1, q2):
    n_states = check_arrays(sv_real, sv_imag)
    _furxy(sv_real, sv_imag, theta, q1, q2, n_states)

def apply_qaoa_furxy_ring(
    sv_real: np.ndarray,
    sv_imag: np.ndarray,
    gammas: typing.Sequence[float],
    betas: typing.Sequence[float],
    hc_diag: np.ndarray,
    n_qubits: int,
    n_trotters: int,
):
    n_states = check_arrays(sv_real, sv_imag, hc_diag)
    n_layers = check_arrays(gammas, betas)
    check_num_qubits(n_qubits, n_states)
    _apply_qaoa_furxy_ring(
        sv_real, sv_imag,
        np.asarray(gammas, dtype="float"),
        np.asarray(betas, dtype="float"),
        hc_diag,
        n_qubits, n_states, n_layers, n_trotters
    )

def apply_qaoa_furxy_complete(
    sv_real: np.ndarray,
    sv_imag: np.ndarray,
    gammas: typing.Sequence[float],
    betas: typing.Sequence[float],
    hc_diag: np.ndarray,
    n_qubits: int,
    n_trotters: int,
):
    n_states = check_arrays(sv_real, sv_imag, hc_diag)
    n_layers = check_arrays(gammas, betas)
    check_num_qubits(n_qubits, n_states)
    _apply_qaoa_furxy_complete(
        sv_real, sv_imag,
        np.asarray(gammas, dtype="float"),
        np.asarray(betas, dtype="float"),
        hc_diag,
        n_qubits, n_states, n_layers, n_trotters
    )
