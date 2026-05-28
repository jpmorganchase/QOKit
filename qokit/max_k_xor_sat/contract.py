###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
"""Multi-backend dispatcher for QAOA tensor contraction on Max-k-XOR-SAT.

Public API:
    contract_symmetric_tree(gammas, betas, p, D, k, precision, verbose, backend)
    contract_with_grad(gammas, betas, p, D, k, precision, backend)
    light_cone_size(p, D, k)
"""

import numpy as np


def _get_cpp_backend():
    """Try to import the C++ backend. Returns module or None."""
    try:
        from qokit.max_k_xor_sat import cpp_backend

        # Check if C++ is actually available (shared lib or CLI)
        cpp_backend._load_lib()
        import os

        if cpp_backend._LIB is not None or os.path.exists(cpp_backend._CLI_BINARY):
            return cpp_backend
    except (ImportError, RuntimeError):
        pass
    return None


def _get_jax_backend():
    """Try to import the JAX backend. Returns module or None."""
    try:
        from qokit.max_k_xor_sat import jax as jax_backend

        if jax_backend.HAS_JAX:
            return jax_backend
    except ImportError:
        pass
    return None


def contract_symmetric_tree(gammas, betas, p, D, k=2, precision="float64", verbose=False, backend="auto"):
    """Compute <Z^{otimes k}> for depth-p QAOA on a D-regular k-uniform hypergraph tree.

    Parameters
    ----------
    gammas, betas : array-like of shape (p,)
        QAOA phase separator and mixer angles.
    p : int
        QAOA depth.
    D : int
        Vertex degree regularity.
    k : int
        Hyperedge arity (default 2 = MaxCut).
    precision : str
        'float64' or 'dd' (double-double, C++ only).
    verbose : bool
        Print progress (C++ backend only).
    backend : str
        'auto' (default), 'cpp', or 'jax'.

    Returns
    -------
    float
        The expectation value <Z^{otimes k}>.
    """
    if backend == "cpp" or backend == "auto":
        cpp = _get_cpp_backend()
        if cpp is not None:
            return cpp.contract_symmetric_tree(gammas, betas, p, D, k, precision=precision, verbose=verbose)
        if backend == "cpp":
            raise RuntimeError("C++ backend not available. Build with:\n" "  cd qokit/max_k_xor_sat/cpp && mkdir -p build && cd build && cmake .. && make -j")

    if backend == "jax" or backend == "auto":
        if precision == "dd":
            raise ValueError("JAX backend does not support double-double precision. Use backend='cpp'.")
        jax_mod = _get_jax_backend()
        if jax_mod is not None:
            return jax_mod.contract_symmetric_tree(gammas, betas, p, D, k)
        if backend == "jax":
            raise ImportError("JAX backend not available. Install with: pip install 'qokit[xorsat-gpu]'")

    raise RuntimeError(
        "No backend available for max_k_xor_sat contraction.\n"
        "Install one of:\n"
        "  - C++ backend: cd qokit/max_k_xor_sat/cpp && mkdir -p build && cd build && cmake .. && make -j\n"
        "  - JAX backend: pip install 'qokit[xorsat-gpu]'"
    )


def contract_with_grad(gammas, betas, p, D, k=2, precision="float64", backend="auto"):
    """Compute <Z^{otimes k}> and its gradient.

    Parameters
    ----------
    gammas, betas : array-like of shape (p,)
        QAOA angles.
    p : int
        QAOA depth.
    D : int
        Vertex degree.
    k : int
        Hyperedge arity (default 2).
    precision : str
        'float64' or 'dd' (C++ only).
    backend : str
        'auto', 'cpp', or 'jax'.

    Returns
    -------
    value : float
        The expectation value <Z^{otimes k}>.
    grad_gammas : ndarray of shape (p,)
        Gradient w.r.t. gammas.
    grad_betas : ndarray of shape (p,)
        Gradient w.r.t. betas.
    """
    if backend == "cpp" or backend == "auto":
        cpp = _get_cpp_backend()
        if cpp is not None:
            return cpp.contract_with_grad(gammas, betas, p, D, k, precision=precision)
        if backend == "cpp":
            raise RuntimeError("C++ backend not available.")

    if backend == "jax" or backend == "auto":
        if precision == "dd":
            raise ValueError("JAX backend does not support double-double precision.")
        jax_mod = _get_jax_backend()
        if jax_mod is not None:
            return jax_mod.contract_with_grad(gammas, betas, p, D, k)
        if backend == "jax":
            raise ImportError("JAX backend not available.")

    raise RuntimeError("No backend available. See contract_symmetric_tree docs for install instructions.")


def light_cone_size(p, D, k=2):
    """Number of qubits in the depth-p light cone.

    N_lc = k * (a^{p+1} - 1) / (a - 1), where a = (D-1)(k-1).

    Parameters
    ----------
    p : int
        QAOA depth.
    D : int
        Vertex degree.
    k : int
        Hyperedge arity (default 2).

    Returns
    -------
    int
        Light cone size (number of qubits).
    """
    a = (D - 1) * (k - 1)
    if a <= 0:
        return k
    if a == 1:
        return k * (p + 1)
    return k * (a ** (p + 1) - 1) // (a - 1)
