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
_BLOCK_DEF = 1024                    # sensible L2-cache block
_EPS_SCALE = 1e-6                    # minimum scale to avoid zeros

# ── encode ───────────────────────────────────────────────────────────────────
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

    if bits <= 8:
        int_max = (1 << (bits - 1)) - 1   # e.g., 127 for 8-bit
        qdtype  = np.int8
    else:
        int_max = (1 << (bits - 1)) - 1   # e.g., 32767 for 16-bit
        qdtype  = np.int16

    rq, iq, scl = [], [], []
    for i in range(0, x.size, block_size):
        blk   = x[i:i + block_size]
        scale = max(_EPS_SCALE, np.abs(blk).max())  # avoid tiny scale
        scl.append(scale)

        r = np.clip(np.round(blk.real / scale * int_max), -int_max, int_max)
        im = np.clip(np.round(blk.imag / scale * int_max), -int_max, int_max)
        rq.append(r.astype(qdtype))
        iq.append(im.astype(qdtype))

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
    renorm: bool    = False,
) -> np.ndarray:
    """
    Inverse of :func:`quantise_fp`.

    Parameters
    ----------
    renorm : bool
        If True  → re-normalise the vector to ‖ψ‖=1.  
        If False → keep raw scale (recommended for benchmarking).
    """
    int_max = (1 << (bits - 1)) - 1
    out     = np.empty(rq.shape, np.complex64)

    idx = 0
    for s in scales:
        sl = slice(idx, idx + block_size)
        idx += block_size

        re = rq[sl].astype(np.float32) / int_max
        im = iq[sl].astype(np.float32) / int_max
        out[sl] = (re + 1j * im) * s

    if renorm:
        out /= np.linalg.norm(out)

    return out
