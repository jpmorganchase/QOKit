###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
"""JAX/GPU backend for QAOA tensor contraction on Max-k-XOR-SAT.

Requires JAX with float64 support. Install with:
    pip install 'qokit[xorsat-gpu]'
"""

try:
    from qokit.max_k_xor_sat.jax.contract import (
        contract_symmetric_tree,
        contract_with_grad,
        light_cone_size,
    )

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
