###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
"""Python ctypes binding for the C++ QAOA contraction library.

Loads the shared library built from the C++ sources and exposes
contract_symmetric_tree() as a Python callable with the same signature
as the public API in qokit.max_k_xor_sat.contract.

Usage:
    from qokit.max_k_xor_sat.cpp_backend import contract_symmetric_tree
    val = contract_symmetric_tree(gammas, betas, p, D, k)

Build the shared library first:
    cd qokit/max_k_xor_sat/cpp && mkdir -p build && cd build && cmake -DBUILD_SHARED_LIBS=ON .. && make -j
"""

import ctypes
import json
import os
import subprocess

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.join(_THIS_DIR, "cpp", "build")
_CLI_BINARY = os.path.join(_BUILD_DIR, "qaoa_contract")


# -- CLI-based wrapper (always works if binary is built) ----------------


def contract_symmetric_tree_cli(gammas, betas, p, D, k=2, precision="float64", verbose=False):
    """Evaluate via the C++ CLI subprocess.

    This has ~10ms overhead per call from process spawn. Suitable for
    high-p evaluations where each contraction takes seconds+, but not
    for low-p optimization where overhead dominates.
    """
    gammas = np.asarray(gammas, dtype=float)
    betas = np.asarray(betas, dtype=float)

    g_str = ",".join(f"{v:.17g}" for v in gammas)
    b_str = ",".join(f"{v:.17g}" for v in betas)

    cmd = [_CLI_BINARY, "--k", str(k), "--D", str(D), "--p", str(p), "--gammas", g_str, "--betas", b_str]
    if precision == "dd":
        cmd.extend(["--precision", "dd"])
    if verbose:
        cmd.append("--verbose")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"C++ CLI failed: {result.stderr}")

    data = json.loads(result.stdout.strip())
    return data["expectation"]


# -- Shared library wrapper (lower overhead) ----------------------------

_LIB = None
_LIB_PATH_CANDIDATES = [
    os.path.join(_BUILD_DIR, "libqaoa_contract.so"),
    os.path.join(_BUILD_DIR, "libqaoa_contract.dylib"),
]


def _load_lib():
    """Try to load the shared library."""
    global _LIB
    if _LIB is not None:
        return _LIB

    for path in _LIB_PATH_CANDIDATES:
        if os.path.exists(path):
            _LIB = ctypes.CDLL(path)
            # Set up function signatures (extern "C" names from exports.cpp)
            _LIB.qaoa_contract.restype = ctypes.c_double
            _LIB.qaoa_contract.argtypes = [
                ctypes.POINTER(ctypes.c_double),  # gammas
                ctypes.POINTER(ctypes.c_double),  # betas
                ctypes.c_int,  # p
                ctypes.c_int,  # D
                ctypes.c_int,  # k
                ctypes.c_int,  # use_dd
                ctypes.c_int,  # verbose
            ]
            _LIB.qaoa_contract_grad.restype = ctypes.c_double
            _LIB.qaoa_contract_grad.argtypes = [
                ctypes.POINTER(ctypes.c_double),  # gammas
                ctypes.POINTER(ctypes.c_double),  # betas
                ctypes.c_int,  # p
                ctypes.c_int,  # D
                ctypes.c_int,  # k
                ctypes.POINTER(ctypes.c_double),  # grad_gammas
                ctypes.POINTER(ctypes.c_double),  # grad_betas
                ctypes.c_int,  # use_dd
            ]
            _LIB.qaoa_light_cone_size.restype = ctypes.c_longlong
            _LIB.qaoa_light_cone_size.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
            return _LIB

    return None


def contract_symmetric_tree_lib(gammas, betas, p, D, k=2, precision="float64", verbose=False):
    """Evaluate via the C++ shared library (ctypes, no subprocess overhead)."""
    lib = _load_lib()
    if lib is None:
        raise RuntimeError("C++ shared library not found. Build with:\n" "  cd qokit/max_k_xor_sat/cpp/build && cmake -DBUILD_SHARED_LIBS=ON .. && make -j")

    gammas = np.asarray(gammas, dtype=np.float64, order="C")
    betas = np.asarray(betas, dtype=np.float64, order="C")

    g_ptr = gammas.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    b_ptr = betas.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    use_dd = precision == "dd"

    return lib.qaoa_contract(g_ptr, b_ptr, p, D, k, int(use_dd), int(verbose))


# -- Public API: auto-select best available method ----------------------


def contract_symmetric_tree(gammas, betas, p, D, k=2, precision="float64", verbose=False):
    """Evaluate <Z^{otimes k}> using the C++ backend.

    Tries the shared library first (zero overhead), falls back to CLI
    (~10ms overhead per call).
    """
    # Try shared library first
    lib = _load_lib()
    if lib is not None:
        return contract_symmetric_tree_lib(gammas, betas, p, D, k, precision=precision, verbose=verbose)

    # Fall back to CLI
    if os.path.exists(_CLI_BINARY):
        return contract_symmetric_tree_cli(gammas, betas, p, D, k, precision=precision, verbose=verbose)

    raise RuntimeError("C++ backend not available. Build with:\n" "  cd qokit/max_k_xor_sat/cpp && mkdir -p build && cd build && cmake .. && make -j")


def _cpp_fd_grad(gammas, betas, p, D, k, use_dd):
    """Call C++ parallel FD gradient via shared library. Returns None on failure."""
    lib = _load_lib()
    if lib is None:
        return None
    g_ptr = gammas.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    b_ptr = betas.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    grad_g = np.empty(p, dtype=np.float64)
    grad_b = np.empty(p, dtype=np.float64)
    gg_ptr = grad_g.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    gb_ptr = grad_b.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    val = lib.qaoa_contract_grad(g_ptr, b_ptr, p, D, k, gg_ptr, gb_ptr, int(use_dd))
    return float(val), grad_g, grad_b


def contract_with_grad(gammas, betas, p, D, k=2, precision="float64"):
    """Compute value + gradient using the best available method.

    When the C++ shared library is available:
    - Float64: uses reverse-mode adjoint (exact, ~3x one eval).
    - DD: uses C++ parallel finite differences.

    Falls back to sequential Python FD when the C++ library is unavailable.
    """
    gammas = np.asarray(gammas, dtype=np.float64, order="C")
    betas = np.asarray(betas, dtype=np.float64, order="C")

    use_dd = precision == "dd"

    # Primary path: C++ parallel FD (works for both float64 and DD).
    result = _cpp_fd_grad(gammas, betas, p, D, k, use_dd)
    if result is not None:
        return result

    # Last resort: sequential FD with adaptive step size.
    a = (D - 1) * (k - 1)
    eps = 1e-32 if precision == "dd" else 2.3e-16
    noise = max(a ** (p / 2.0), 1.0) * eps if a > 1 else eps
    noise = max(noise, 1e-30 if precision == "dd" else 1e-14)
    h = max(min(noise ** (1.0 / 3.0), 1e-2), 1e-8)
    val = contract_symmetric_tree(gammas, betas, p, D, k, precision=precision)
    grad_g = np.empty(p)
    grad_b = np.empty(p)
    for i in range(p):
        g_p = gammas.copy()
        g_p[i] += h
        g_m = gammas.copy()
        g_m[i] -= h
        grad_g[i] = (contract_symmetric_tree(g_p, betas, p, D, k, precision=precision) - contract_symmetric_tree(g_m, betas, p, D, k, precision=precision)) / (
            2 * h
        )
    for i in range(p):
        b_p = betas.copy()
        b_p[i] += h
        b_m = betas.copy()
        b_m[i] -= h
        grad_b[i] = (
            contract_symmetric_tree(gammas, b_p, p, D, k, precision=precision) - contract_symmetric_tree(gammas, b_m, p, D, k, precision=precision)
        ) / (2 * h)
    return float(val), grad_g, grad_b


def light_cone_size(p, D, k):
    """Light cone size via C++ (or Python fallback)."""
    lib = _load_lib()
    if lib is not None:
        return lib.qaoa_light_cone_size(p, D, k)
    a = (D - 1) * (k - 1)
    if a <= 0:
        return k
    if a == 1:
        return k * (p + 1)
    return k * (a ** (p + 1) - 1) // (a - 1)
