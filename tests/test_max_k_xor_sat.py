###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
"""Tests for the max_k_xor_sat module."""
import numpy as np
import pytest

from qokit.max_k_xor_sat import (
    contract_symmetric_tree,
    light_cone_size,
    load_benchmark_energies,
    load_precomputed_results,
    get_available_configs,
)

# -- Backend availability flags --

try:
    from qokit.max_k_xor_sat.cpp_backend import contract_symmetric_tree as cpp_contract
    from qokit.max_k_xor_sat.cpp_backend import _load_lib
    import os
    from qokit.max_k_xor_sat.cpp_backend import _CLI_BINARY

    _load_lib()
    from qokit.max_k_xor_sat.cpp_backend import _LIB

    HAS_CPP = _LIB is not None or os.path.exists(_CLI_BINARY)
except (ImportError, RuntimeError):
    HAS_CPP = False

try:
    import jax

    jax.config.update("jax_enable_x64", True)
    from qokit.max_k_xor_sat.jax import contract_symmetric_tree as jax_contract, HAS_JAX
except (ImportError, RuntimeError):
    HAS_JAX = False

try:
    import pybobyqa  # noqa: F401
    from qokit.max_k_xor_sat.optimize import optimize_angles

    HAS_OPTIMIZE = True
except ImportError:
    HAS_OPTIMIZE = False

try:
    import quimb.tensor as qtn

    HAS_QUIMB = True
except ImportError:
    HAS_QUIMB = False


# -- Test parameters --

CASES = [
    (2, 2, 1),
    (2, 2, 2),
    (2, 2, 3),
    (2, 3, 1),
    (2, 3, 2),
    (2, 3, 3),
    (3, 2, 1),
    (3, 2, 2),
    (3, 3, 1),
    (3, 3, 2),
    (4, 2, 1),
    (4, 3, 1),
]

HIGH_P_REFS = {
    (2, 3, 8): 0.47718413498329193,
    (2, 3, 10): 0.4916066388958419,
    (2, 3, 12): 0.4279034805223548,
}


def _params(p):
    """Deterministic test parameters for depth p."""
    return (
        np.array([0.3 + 0.1 * i for i in range(p)]),
        np.array([0.4 - 0.05 * i for i in range(p)]),
    )


# -- Quimb reference (test-only) --


def quimb_exact(gammas, betas, p, D, k):
    """Exact <Z^k> via quimb full state-vector — ground truth for testing."""
    gammas = np.asarray(gammas, dtype=float)
    betas = np.asarray(betas, dtype=float)

    # Build the light cone tree
    qid = 0
    layers = [list(range(qid, qid + k))]
    qid += k
    root_ids = layers[0][:]
    all_edges = [layers[0][:]]
    for l in range(p):
        next_layer = []
        for v in layers[l]:
            for _ in range(D - 1):
                children = list(range(qid, qid + k - 1))
                qid += k - 1
                next_layer.extend(children)
                all_edges.append([v] + children)
        layers.append(next_layer)
    N = qid

    # k-body phase gate
    dim = 1 << k

    def _zk_gate(gamma):
        U = np.zeros((dim, dim), dtype=complex)
        for x in range(dim):
            U[x, x] = np.exp(-1j * gamma * ((-1) ** (bin(x).count("1") % 2)))
        return U

    # Z^k measurement operator
    Zk = np.zeros((dim, dim), dtype=complex)
    for x in range(dim):
        Zk[x, x] = (-1.0) ** (bin(x).count("1") % 2)

    # Build circuit
    circ = qtn.Circuit(N)
    for q in range(N):
        circ.apply_gate("H", q)
    for ell in range(p):
        U = _zk_gate(gammas[ell])
        for edge in all_edges:
            circ.apply_gate_raw(U, tuple(edge))
        for q in range(N):
            circ.apply_gate("RX", 2 * betas[ell], q)

    return float(np.real(circ.local_expectation(Zk, root_ids, optimize="auto-hq")))


# ============================================================
# Asset tests (always run, no optional deps)
# ============================================================


def test_load_precomputed_results():
    """Verify precomputed results load for a known config."""
    data = load_precomputed_results(3, 5)
    assert data["k"] == 3
    assert data["D"] == 5
    assert "results" in data
    assert "1" in data["results"]
    assert "16" in data["results"]


def test_load_all_precomputed_results():
    """Verify all 16 configs load successfully."""
    configs = get_available_configs()
    assert len(configs) >= 16
    for k, D in configs:
        data = load_precomputed_results(k, D)
        assert data["k"] == k
        assert data["D"] == D


def test_load_benchmark_energies():
    """Verify benchmark energies load."""
    data = load_benchmark_energies()
    assert "columns" in data
    assert "data" in data
    assert len(data["columns"]) == 4
    assert "3,5" in data["data"]


def test_precomputed_results_structure():
    """Verify the structure of a results entry."""
    data = load_precomputed_results(3, 5)
    entry = data["results"]["5"]
    assert "gammas" in entry
    assert "betas" in entry
    assert "expectation" in entry
    assert "objective" in entry
    assert len(entry["gammas"]) == 5
    assert len(entry["betas"]) == 5
    assert 0 < entry["objective"] < 1


def test_light_cone_size():
    """Verify light_cone_size formula: N_lc = k*(a^{p+1}-1)/(a-1), a=(D-1)*(k-1)."""
    # k=2, D=3, p=1: a=2, N=2*(4-1)/(2-1)=6
    assert light_cone_size(1, 3, 2) == 6
    # k=2, D=3, p=2: a=2, N=2*(8-1)/(2-1)=14
    assert light_cone_size(2, 3, 2) == 14
    # k=3, D=2, p=1: a=2, N=3*(4-1)/(2-1)=9
    assert light_cone_size(1, 2, 3) == 9
    # k=2, D=2, p=3: a=1, N=2*(3+1)=8
    assert light_cone_size(3, 2, 2) == 8


def test_load_nonexistent_config():
    """Verify FileNotFoundError for missing config."""
    with pytest.raises(FileNotFoundError):
        load_precomputed_results(99, 99)


# ============================================================
# C++ backend tests
# ============================================================


@pytest.mark.skipif(not HAS_CPP, reason="C++ backend not built")
@pytest.mark.skipif(not HAS_QUIMB, reason="quimb not installed")
@pytest.mark.parametrize("k,D,p", CASES)
def test_cpp_contract_vs_quimb(k, D, p):
    """C++ backend matches quimb exact reference."""
    gammas, betas = _params(p)
    result = cpp_contract(gammas, betas, p, D, k)
    expected = quimb_exact(gammas, betas, p, D, k)
    assert abs(result - expected) < 1e-8, f"k={k} D={D} p={p}: got {result}, expected {expected}"


@pytest.mark.skipif(not HAS_CPP, reason="C++ backend not built")
@pytest.mark.parametrize("key,expected", HIGH_P_REFS.items())
def test_cpp_high_p_regression(key, expected):
    """C++ backend matches high-p reference values."""
    k, D, p = key
    gammas, betas = _params(p)
    result = cpp_contract(gammas, betas, p, D, k)
    assert abs(result - expected) < 1e-8, f"k={k} D={D} p={p}: got {result}, expected {expected}"


# ============================================================
# JAX backend tests
# ============================================================


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
@pytest.mark.skipif(not HAS_QUIMB, reason="quimb not installed")
@pytest.mark.parametrize("k,D,p", CASES)
def test_jax_contract_vs_quimb(k, D, p):
    """JAX backend matches quimb exact reference."""
    gammas, betas = _params(p)
    result = jax_contract(gammas, betas, p, D, k)
    expected = quimb_exact(gammas, betas, p, D, k)
    assert abs(result - expected) < 1e-8, f"k={k} D={D} p={p}: got {result}, expected {expected}"


# ============================================================
# Cross-backend consistency
# ============================================================


@pytest.mark.skipif(not (HAS_CPP and HAS_JAX), reason="Need both C++ and JAX backends")
@pytest.mark.parametrize("k,D,p", CASES[:6])
def test_cpp_jax_agreement(k, D, p):
    """C++ and JAX backends produce the same result."""
    gammas, betas = _params(p)
    cpp_val = cpp_contract(gammas, betas, p, D, k)
    jax_val = jax_contract(gammas, betas, p, D, k)
    assert abs(cpp_val - jax_val) < 1e-10, f"k={k} D={D} p={p}: cpp={cpp_val}, jax={jax_val}"


# ============================================================
# Optimizer tests
# ============================================================


@pytest.mark.skipif(not HAS_OPTIMIZE, reason="Optimizer dependencies not installed")
@pytest.mark.skipif(not (HAS_CPP or HAS_JAX), reason="No backend available")
def test_optimize_returns_structure():
    """Optimizer returns expected dict structure."""
    result = optimize_angles(k=2, D=3, p=2, maxiter=10, verbose=False)
    assert "gammas" in result
    assert "betas" in result
    assert "expectation" in result
    assert "objective" in result
    assert "num_evals" in result
    assert "converged" in result
    assert "seed_source" in result
    assert len(result["gammas"]) == 2
    assert len(result["betas"]) == 2


def test_chebyshev_interp_identity():
    """Chebyshev interp with same source and target depth is identity."""
    from qokit.max_k_xor_sat.optimize.seed import chebyshev_interp

    angles = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    result = chebyshev_interp(angles, 5, 5)
    np.testing.assert_allclose(result, angles, atol=1e-12)


def test_chebyshev_extrap_shape():
    """Chebyshev extrap produces correct output shape."""
    from qokit.max_k_xor_sat.optimize.seed import chebyshev_extrap

    angles = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    result = chebyshev_extrap(angles, 5, 8)
    assert result.shape == (8,)
    assert result[0] == angles[0]  # first element pinned


def test_seed_loading():
    """Seed loading finds precomputed results."""
    from qokit.max_k_xor_sat.optimize.seed import load_seed_angles

    gammas, betas, source = load_seed_angles(3, 5, 5)
    assert len(gammas) == 5
    assert len(betas) == 5
    assert "results" in source


# ============================================================
# FD helper for gradient accuracy tests
# ============================================================

_GRAD_CASES = [(2, 3, 1), (2, 3, 2), (3, 2, 1), (3, 3, 1)]


def _fd_grad(contract_fn, gammas, betas, p, D, k=2, h=1e-5):
    """Numerical central finite-difference gradient of contract_fn."""
    gammas = np.asarray(gammas, dtype=float)
    betas = np.asarray(betas, dtype=float)
    grad_g = np.empty(p)
    grad_b = np.empty(p)
    for i in range(p):
        g_p = gammas.copy()
        g_p[i] += h
        g_m = gammas.copy()
        g_m[i] -= h
        grad_g[i] = (contract_fn(g_p, betas, p, D, k) - contract_fn(g_m, betas, p, D, k)) / (2 * h)
    for i in range(p):
        b_p = betas.copy()
        b_p[i] += h
        b_m = betas.copy()
        b_m[i] -= h
        grad_b[i] = (contract_fn(gammas, b_p, p, D, k) - contract_fn(gammas, b_m, p, D, k)) / (2 * h)
    return grad_g, grad_b


# ============================================================
# contract.py dispatcher tests
# ============================================================


@pytest.mark.skipif(not (HAS_CPP or HAS_JAX), reason="No backend available")
@pytest.mark.parametrize("k,D,p", CASES[:6])
def test_dispatcher_contract_symmetric_tree(k, D, p):
    """Dispatcher contract_symmetric_tree auto-selects a backend and returns a valid float."""
    from qokit.max_k_xor_sat.contract import contract_symmetric_tree as disp_contract

    gammas, betas = _params(p)
    val = disp_contract(gammas, betas, p, D, k)
    assert isinstance(val, float)
    assert -1.0 <= val <= 1.0


@pytest.mark.skipif(not HAS_CPP, reason="C++ backend not built")
def test_dispatcher_contract_symmetric_tree_cpp():
    """Dispatcher backend='cpp' matches cpp_backend directly."""
    from qokit.max_k_xor_sat.contract import contract_symmetric_tree as disp_contract

    gammas, betas = _params(2)
    val_disp = disp_contract(gammas, betas, 2, 3, k=2, backend="cpp")
    val_cpp = cpp_contract(gammas, betas, 2, 3, 2)
    assert abs(val_disp - val_cpp) < 1e-12


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
def test_dispatcher_contract_symmetric_tree_jax():
    """Dispatcher backend='jax' matches jax backend directly."""
    from qokit.max_k_xor_sat.contract import contract_symmetric_tree as disp_contract

    gammas, betas = _params(2)
    val_disp = disp_contract(gammas, betas, 2, 3, k=2, backend="jax")
    val_jax = jax_contract(gammas, betas, 2, 3, 2)
    assert abs(val_disp - val_jax) < 1e-12


def test_dispatcher_jax_dd_raises():
    """Dispatcher raises ValueError when backend='jax' + precision='dd' requested."""
    from qokit.max_k_xor_sat.contract import contract_symmetric_tree as disp_contract

    gammas, betas = _params(2)
    with pytest.raises(ValueError, match="double-double"):
        disp_contract(gammas, betas, 2, 3, k=2, precision="dd", backend="jax")


@pytest.mark.skipif(not (HAS_CPP or HAS_JAX), reason="No backend available")
@pytest.mark.parametrize("k,D,p", _GRAD_CASES)
def test_dispatcher_contract_with_grad_shape(k, D, p):
    """Dispatcher contract_with_grad returns (float, array(p,), array(p,))."""
    from qokit.max_k_xor_sat.contract import contract_with_grad as disp_grad

    gammas, betas = _params(p)
    val, dg, db = disp_grad(gammas, betas, p, D, k)
    assert isinstance(val, float)
    assert np.asarray(dg).shape == (p,)
    assert np.asarray(db).shape == (p,)


@pytest.mark.skipif(not (HAS_CPP or HAS_JAX), reason="No backend available")
@pytest.mark.parametrize("k,D,p", _GRAD_CASES)
def test_dispatcher_contract_with_grad_value(k, D, p):
    """Dispatcher contract_with_grad value matches contract_symmetric_tree."""
    from qokit.max_k_xor_sat.contract import (
        contract_symmetric_tree as disp_contract,
        contract_with_grad as disp_grad,
    )

    gammas, betas = _params(p)
    val_tree = disp_contract(gammas, betas, p, D, k)
    val_grad, _, _ = disp_grad(gammas, betas, p, D, k)
    assert abs(val_tree - val_grad) < 1e-8, f"k={k} D={D} p={p}: tree={val_tree} grad={val_grad}"


@pytest.mark.skipif(not (HAS_CPP or HAS_JAX), reason="No backend available")
@pytest.mark.parametrize("k,D,p", _GRAD_CASES)
def test_dispatcher_contract_with_grad_accuracy(k, D, p):
    """Dispatcher contract_with_grad gradient matches finite differences."""
    from qokit.max_k_xor_sat.contract import (
        contract_symmetric_tree as disp_contract,
        contract_with_grad as disp_grad,
    )

    gammas, betas = _params(p)
    _, dg, db = disp_grad(gammas, betas, p, D, k)
    ref_dg, ref_db = _fd_grad(disp_contract, gammas, betas, p, D, k)
    np.testing.assert_allclose(dg, ref_dg, atol=1e-4, rtol=1e-4, err_msg=f"gamma grad k={k} D={D} p={p}")
    np.testing.assert_allclose(db, ref_db, atol=1e-4, rtol=1e-4, err_msg=f"beta grad k={k} D={D} p={p}")


# ============================================================
# cpp_backend.py additional tests
# ============================================================


@pytest.mark.skipif(not HAS_CPP, reason="C++ backend not built")
@pytest.mark.parametrize("k,D,p", [(2, 3, 1), (2, 3, 2), (3, 2, 1), (2, 2, 3)])
def test_cpp_light_cone_size(k, D, p):
    """cpp_backend light_cone_size matches reference formula."""
    from qokit.max_k_xor_sat.cpp_backend import light_cone_size as cpp_lcs
    from qokit.max_k_xor_sat.contract import light_cone_size as ref_lcs

    assert cpp_lcs(p, D, k) == ref_lcs(p, D, k), f"k={k} D={D} p={p}"


@pytest.mark.skipif(not HAS_CPP, reason="C++ backend not built")
@pytest.mark.parametrize("k,D,p", _GRAD_CASES)
def test_cpp_contract_with_grad_shape(k, D, p):
    """cpp_backend contract_with_grad returns (float, array(p,), array(p,))."""
    from qokit.max_k_xor_sat.cpp_backend import contract_with_grad as cpp_grad

    gammas, betas = _params(p)
    val, dg, db = cpp_grad(gammas, betas, p, D, k)
    assert isinstance(val, float)
    assert np.asarray(dg).shape == (p,)
    assert np.asarray(db).shape == (p,)


@pytest.mark.skipif(not HAS_CPP, reason="C++ backend not built")
@pytest.mark.parametrize("k,D,p", _GRAD_CASES)
def test_cpp_contract_with_grad_value(k, D, p):
    """cpp_backend contract_with_grad value matches contract_symmetric_tree."""
    from qokit.max_k_xor_sat.cpp_backend import contract_with_grad as cpp_grad

    gammas, betas = _params(p)
    val_tree = cpp_contract(gammas, betas, p, D, k)
    val_grad, _, _ = cpp_grad(gammas, betas, p, D, k)
    assert abs(val_tree - val_grad) < 1e-8, f"k={k} D={D} p={p}: tree={val_tree} grad={val_grad}"


@pytest.mark.skipif(not HAS_CPP, reason="C++ backend not built")
@pytest.mark.parametrize("k,D,p", _GRAD_CASES)
def test_cpp_grad_accuracy(k, D, p):
    """cpp_backend gradient matches finite differences."""
    from qokit.max_k_xor_sat.cpp_backend import contract_with_grad as cpp_grad

    gammas, betas = _params(p)
    _, dg, db = cpp_grad(gammas, betas, p, D, k)
    ref_dg, ref_db = _fd_grad(cpp_contract, gammas, betas, p, D, k)
    np.testing.assert_allclose(dg, ref_dg, atol=1e-4, rtol=1e-4, err_msg=f"gamma grad k={k} D={D} p={p}")
    np.testing.assert_allclose(db, ref_db, atol=1e-4, rtol=1e-4, err_msg=f"beta grad k={k} D={D} p={p}")


# ============================================================
# JAX backend additional tests
# ============================================================


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
@pytest.mark.parametrize("k,D,p", [(2, 3, 1), (2, 3, 2), (3, 2, 1), (2, 2, 3)])
def test_jax_light_cone_size(k, D, p):
    """JAX backend light_cone_size matches reference formula."""
    from qokit.max_k_xor_sat.jax.contract import light_cone_size as jax_lcs
    from qokit.max_k_xor_sat.contract import light_cone_size as ref_lcs

    assert jax_lcs(p, D, k) == ref_lcs(p, D, k), f"k={k} D={D} p={p}"


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
@pytest.mark.parametrize("k,D,p", _GRAD_CASES)
def test_jax_contract_with_grad_shape(k, D, p):
    """JAX backend contract_with_grad returns (float, array(p,), array(p,))."""
    from qokit.max_k_xor_sat.jax import contract_with_grad as jax_grad

    gammas, betas = _params(p)
    val, dg, db = jax_grad(gammas, betas, p, D, k)
    assert isinstance(val, float)
    assert np.asarray(dg).shape == (p,)
    assert np.asarray(db).shape == (p,)


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
@pytest.mark.parametrize("k,D,p", _GRAD_CASES)
def test_jax_contract_with_grad_value(k, D, p):
    """JAX backend contract_with_grad value matches contract_symmetric_tree."""
    from qokit.max_k_xor_sat.jax import contract_with_grad as jax_grad

    gammas, betas = _params(p)
    val_tree = jax_contract(gammas, betas, p, D, k)
    val_grad, _, _ = jax_grad(gammas, betas, p, D, k)
    assert abs(val_tree - val_grad) < 1e-8, f"k={k} D={D} p={p}: tree={val_tree} grad={val_grad}"


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
@pytest.mark.parametrize("k,D,p", _GRAD_CASES)
def test_jax_grad_accuracy(k, D, p):
    """JAX backend gradient matches finite differences."""
    from qokit.max_k_xor_sat.jax import contract_with_grad as jax_grad

    gammas, betas = _params(p)
    _, dg, db = jax_grad(gammas, betas, p, D, k)
    ref_dg, ref_db = _fd_grad(jax_contract, gammas, betas, p, D, k)
    np.testing.assert_allclose(dg, ref_dg, atol=1e-4, rtol=1e-4, err_msg=f"gamma grad k={k} D={D} p={p}")
    np.testing.assert_allclose(db, ref_db, atol=1e-4, rtol=1e-4, err_msg=f"beta grad k={k} D={D} p={p}")
