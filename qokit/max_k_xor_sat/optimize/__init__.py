###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
"""QAOA angle optimizer with Chebyshev warm-start.

Optimizes QAOA angles (gammas, betas) for a given (k, D, p) by maximizing
the energy (1 - <Z^k>)/2 via symmetric tensor contraction.

Usage (programmatic):
    from qokit.max_k_xor_sat.optimize import optimize_angles, chebyshev_interp
    result = optimize_angles(k=2, D=3, p=3)
"""

# Contraction function exposed at package level so that callers
# can patch it for an alternative backend:
#   import qokit.max_k_xor_sat.optimize as _opt_mod
#   _opt_mod.contract_symmetric_tree = jax_fn
from qokit.max_k_xor_sat.contract import contract_symmetric_tree

# Gradient function: patchable at package level, lazily imported.
_contract_with_grad = None

from qokit.max_k_xor_sat.optimize.seed import (
    angles_to_cheb,
    cheb_basis_matrix,
    cheb_to_angles,
    chebyshev_extrap,
    chebyshev_interp,
    load_output_file,
    load_seed_angles,
    save_output_file,
    scaled_extrap,
    _RESOURCES_DIR,
)

from qokit.max_k_xor_sat.optimize.core import optimize_angles
