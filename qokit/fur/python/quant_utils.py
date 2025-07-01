###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
"""
Light-weight fixed-point (FP) quantisation helpers – **Python-only**.
Used by the patched FUR simulators when the user passes
``quant_bits=…`` to ``simulate_qaoa``.
"""

from __future__ import annotations
import numpy as np

# ── constants ────────────────────────────────────────────────────────────────
_INT8_MAX  = (1 << 7)  - 1          #  127
_INT16_MAX = (1 << 15) - 1          # 32767
_BLOCK_DEF = 1024                   # sensible L2-cache block

# ── encode ────────────────────────────────────────────────────────────
def quantise_fp(
    x: np.ndarray,
    *,
    bits: int = 8,
    block_size: int = _BLOCK_DEF,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Symmetric mid-tread fixed-point quantiser (real / imag handled separately)
    returning rq, iq, scales
    """
    x = np.asarray(x, np.complex64)

    # --- NEW: choose range & dtype according to *bits* -----------------
    if bits <= 8:
        int_max = (1 << (bits - 1)) - 1   #  7, 31, 127 for 4/6/8-bit
        qdtype  = np.int8
    else:
        int_max = (1 << (bits - 1)) - 1   # 32767 for 16-bit, …
        qdtype  = np.int16
    # ------------------------------------------------------------------

    rq, iq, scl = [], [], []
    for i in range(0, x.size, block_size):
        blk   = x[i:i + block_size]
        scale = max(1e-12, np.abs(blk).max())
        scl.append(scale)
        rq.append(np.round(blk.real / scale * int_max).astype(qdtype))
        iq.append(np.round(blk.imag / scale * int_max).astype(qdtype))

    return (np.concatenate(rq),
            np.concatenate(iq),
            np.asarray(scl, np.float32))


# ── decode ───────────────────────────────────────────────────────────────────
def dequantise_fp(
    rq: np.ndarray,
    iq: np.ndarray,
    scales: np.ndarray,
    *,
    bits: int       = 8,
    block_size: int = _BLOCK_DEF,
    renorm: bool    = False,          # ← NEW flag (default = keep raw error)
) -> np.ndarray:
    """
    Inverse of :func:`quantise_fp`.

    Parameters
    ----------
    renorm : bool
        If True  → re-normalise the vector to ‖ψ‖=1 (old behaviour).  
        If False → leave amplitudes as-is so the quantisation error is
        preserved (recommended for benchmarking).
    """
    int_max = (1 << (bits - 1)) - 1      # 7, 31, 127, 32 767 … exactly
    out     = np.empty(rq.shape, np.complex64)

    idx = 0
    for s in scales:                           # per-block reconstruction
        sl       = slice(idx, idx + block_size); idx += block_size
        out[sl]  = (rq[sl].astype(np.float32) / int_max +
                    1j * iq[sl].astype(np.float32) / int_max) * s

    if renorm:                                 # OPTIONAL normalisation
        out /= np.linalg.norm(out)

    return out

