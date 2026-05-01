###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
"""QAOA on Max-k-XOR-SAT via symmetric tensor contraction.

Computes <Z^{otimes k}> at depth p on D-regular, k-uniform hypergraph trees.
Cost O(p * 4^p), independent of D, k, and light-cone size N_lc.

Backends:
  - C++ (default): float64 + double-double precision, cmake build required.
  - JAX/GPU: float64 only, install with pip install 'qokit[xorsat-gpu]'.

Example:
    from qokit.max_k_xor_sat import contract_symmetric_tree, light_cone_size
    val = contract_symmetric_tree(gammas, betas, p=5, D=3, k=2)
"""

import json
import os

from qokit.max_k_xor_sat.contract import (
    contract_symmetric_tree,
    contract_with_grad,
    light_cone_size,
)

_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def load_precomputed_results(k, D):
    """Load precomputed optimization results for a given (k, D) configuration.

    Parameters
    ----------
    k : int
        Hyperedge arity.
    D : int
        Vertex degree.

    Returns
    -------
    dict
        Parsed JSON with keys: 'k', 'D', 'convention', 'results'.
        results[str(p)] has: gammas, betas, expectation, objective, num_evals,
        converged, seed_source.

    Raises
    ------
    FileNotFoundError
        If no results file exists for the given (k, D).
    """
    path = os.path.join(_ASSETS_DIR, f"results_k{k}_D{D}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No precomputed results for k={k}, D={D}. " f"Available files: {_list_available_configs()}")
    with open(path) as f:
        return json.load(f)


def load_benchmark_energies():
    """Load benchmark energies from classical algorithms.

    Returns
    -------
    dict
        Keys: 'columns' (list of algorithm names),
              'data' (dict mapping 'k,D' -> list of energies).
    """
    path = os.path.join(_ASSETS_DIR, "benchmark_energies.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"benchmark_energies.json not found in {_ASSETS_DIR}")
    with open(path) as f:
        return json.load(f)


def _list_available_configs():
    """List available (k, D) configurations from assets."""
    import re

    pattern = re.compile(r"^results_k(\d+)_D(\d+)\.json$")
    configs = []
    for name in sorted(os.listdir(_ASSETS_DIR)):
        m = pattern.match(name)
        if m:
            configs.append((int(m.group(1)), int(m.group(2))))
    return configs


def get_available_configs():
    """Return list of (k, D) tuples for which precomputed results exist.

    Returns
    -------
    list of tuple
        Available (k, D) configurations.
    """
    return _list_available_configs()


# Lazy imports for optional dependencies
def __getattr__(name):
    if name == "optimize_angles":
        from qokit.max_k_xor_sat.optimize.core import optimize_angles

        return optimize_angles
    if name == "chebyshev_interp":
        from qokit.max_k_xor_sat.optimize.seed import chebyshev_interp

        return chebyshev_interp
    if name == "chebyshev_extrap":
        from qokit.max_k_xor_sat.optimize.seed import chebyshev_extrap

        return chebyshev_extrap
    raise AttributeError(f"module 'qokit.max_k_xor_sat' has no attribute {name!r}")
