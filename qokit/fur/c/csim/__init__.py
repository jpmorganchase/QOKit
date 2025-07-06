# ── NEW: import the int-kernels when the shared library was compiled with
#         -DUSE_INT_KERNELS (see Makefile change).  They will be no-ops if the
#         symbol is missing, so this file works on old builds too.
try:
    from .wrapper import (
        furx_int8,
        apply_qaoa_furx_int8,
         _apply_qaoa_furx_int, 
        furxy_int8,
        apply_qaoa_furxy_ring_int8,
        apply_qaoa_furxy_complete_int8,
    )

    _EXTRA = [
        "furx_int8",
        "_apply_qaoa_furx_int",
        "apply_qaoa_furx_int8",
        "furxy_int8",
        "apply_qaoa_furxy_ring_int8",
        "apply_qaoa_furxy_complete_int8",
    ]
except ImportError:
    # library built without USE_INT_KERNELS → silently ignore
    _EXTRA = []
# ─────────────────────────────────────────────────────────────────────────────

from .wrapper import (
    furx,
    apply_qaoa_furx,
    furxy,
    apply_qaoa_furxy_ring,
    apply_qaoa_furxy_complete,
    _apply_qaoa_furx_int,
)
from .libpath import is_available

__all__ = [
    "furx",
    "apply_qaoa_furx",
    "furxy",
    "apply_qaoa_furxy_ring",
    "apply_qaoa_furxy_complete",
    "is_available",
    "_apply_qaoa_furx_int",
] + _EXTRA        # ← export the extra symbols if present
