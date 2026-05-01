###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
"""Pure JAX/GPU tensor contraction for QAOA on symmetric trees.

All operations use jax.numpy — zero NumPy in the hot path.
Designed for @jax.jit compilation with static_argnums for (p, D, k).
Uses jax.grad() for automatic differentiation.

Cost: O(p * 4^p), independent of D, k, and N_lc.
"""

import functools

import jax
import jax.numpy as jnp

from qokit.max_k_xor_sat.jax.primitives import (
    CHARGE_DIAG,
    doubled_mixer,
    charge_weight_matrix,
    root_charge_weights,
    wht_charge_contract,
    pow_precise,
)


def _hyperedge_branch(gammas, betas, num_rounds, k, child_branch=None):
    """Branch tensor for one hyperedge (4^num_rounds entries).

    Uses the rank-4 decomposition of the k-body phase gate to work
    per-child instead of jointly tracking all k-1 children.

    Parameters
    ----------
    gammas, betas : jnp.array
        Full arrays of p angles.
    num_rounds : int
        Number of rounds for this branch level.
    k : int
        Hyperedge size.
    child_branch : jnp.array or None
        Child branch tensor (4^{num_rounds-1} entries), or None for leaves.

    Returns
    -------
    jnp.array
        Branch tensor, 4^num_rounds entries (flat).
    """
    m = k - 1  # children per hyperedge

    # Precompute modified mixer matrices: MD[ell][a] = M(beta_ell) * CDIAG[a]
    MD = []
    for ell in range(num_rounds):
        M = doubled_mixer(betas[ell])
        MD.append([M * CHARGE_DIAG[a] for a in range(4)])

    # Charge weight matrices
    W = [charge_weight_matrix(gammas[ell]) for ell in range(num_rounds)]

    child_rounds = num_rounds - 1 if child_branch is not None else 0

    # -- Phase 1: coupled contractions consuming child branch --
    if child_branch is not None and child_rounds >= 2:
        V = 0.5 * child_branch.ravel().reshape(1, -1)
        n_ch = 1
        for ell in range(child_rounds - 1):
            T_r = V.reshape(n_ch, 4, 4, -1)
            channels = wht_charge_contract(doubled_mixer(betas[ell]), T_r)
            V = jnp.concatenate([ch.reshape(n_ch, -1) for ch in channels], axis=0)
            n_ch *= 4
        V = V.reshape(-1, 4)
    elif child_branch is not None:
        V = (0.5 * child_branch).reshape(1, 4)
    else:
        V = jnp.full((1, 4), 0.5, dtype=jnp.complex128)

    # -- Phase 2 fused with trace --
    start_mv = max(child_rounds - 1, 0)

    # Trace matrix: columns are K_a[0,:] + K_a[3,:] for a=0..3
    trace_vecs = [MD[num_rounds - 1][a][0, :] + MD[num_rounds - 1][a][3, :] for a in range(4)]
    trace_matrix = jnp.stack(trace_vecs, axis=1)  # (4, 4)

    # Recursive Phase 2 — Python loop unrolls at JIT trace time since
    # num_rounds is static. Dynamic shapes (4x growth) prevent fori_loop.
    def _phase2_trace(V, ell):
        if ell == num_rounds - 1:
            result = V @ trace_matrix  # (n, 4)
            return jnp.concatenate([result[:, a] for a in range(4)])
        parts = [_phase2_trace(V @ MD[ell][a].T, ell + 1) for a in range(4)]
        return jnp.concatenate(parts)

    t = _phase2_trace(V, start_mv)
    t = t.reshape((4,) * num_rounds)

    # Reorder axes
    remaining = num_rounds - start_mv
    if num_rounds > 1:
        perm = list(range(num_rounds - 1, remaining - 1, -1)) + list(range(remaining))
        t = jnp.transpose(t, perm)

    # Entrywise (k-1) power with normalization
    if m > 1:
        t_max = jnp.max(jnp.abs(t))
        # Avoid division by zero — jnp.where is JIT-safe
        t = jnp.where(t_max > 0, t / t_max, t)
    F = pow_precise(t, m).ravel()

    # Mode products: F[h_0,...,h_{r-1}] = sum_a prod_ell W[ell][h_ell,a_ell] * t^m[a]
    # moveaxis is O(1) on GPU (just metadata change)
    F = F.reshape((4,) * num_rounds)
    for ell in range(num_rounds):
        F = jnp.moveaxis(F, ell, 0)
        shape = F.shape
        F = W[ell] @ F.reshape(4, -1)
        F = F.reshape((4,) + shape[1:])
        F = jnp.moveaxis(F, 0, ell)
    F = F.ravel()

    return F


def _make_root_round(ell, R, k):
    """Create a checkpointed root contraction round with static ell/R/k."""

    @jax.checkpoint
    def _root_round(gammas, betas, factor, coeffs):
        M = doubled_mixer(betas[ell])
        u = root_charge_weights(gammas[ell])
        fi = factor.reshape(R, 4, 4, -1)
        channels = wht_charge_contract(M, fi)
        coeff_parts = [u[a] * coeffs for a in range(4)]
        factor_parts = [channels[a].reshape(R, -1) for a in range(4)]
        return jnp.concatenate(factor_parts, axis=0), jnp.concatenate(coeff_parts)

    return _root_round


def _root_contract(rb, gammas, betas, p, D, k):
    """Root contraction using factored rank-1 representation.

    Tracks a single factor instead of k copies (all k root qubits carry
    identical branches). Final round + Z^{otimes k} measurement fused via
    per-qubit z^k power.

    Each intermediate round is checkpointed to limit GPU memory.

    Parameters
    ----------
    rb : jnp.array
        Branch tensor raised to (D-1) power, 4^p entries (flat).
    gammas, betas : jnp.array
        Full angle arrays.
    p, D, k : int
        Problem parameters.

    Returns
    -------
    scalar (float)
        The expectation value <Z^{otimes k}>.
    """
    # Intermediate rounds — each checkpointed
    coeffs = jnp.array([0.5**k], dtype=jnp.complex128)
    factor = rb.reshape(1, -1)
    R = 1

    for ell in range(p - 1):
        root_fn = _make_root_round(ell, R, k)
        factor, coeffs = root_fn(gammas, betas, factor, coeffs)
        R *= 4

    # Final round + Z measurement (per-qubit k-th power)
    M = doubled_mixer(betas[p - 1])
    u = root_charge_weights(gammas[p - 1])

    result = jnp.complex128(0.0)
    for a in range(4):
        K = M * CHARGE_DIAG[a]
        tv = K[0, :] - K[3, :]  # Z trace vector, shape (4,)
        z = factor @ tv  # per-qubit scalars, shape (R,)
        result = result + u[a] * jnp.sum(coeffs * pow_precise(z, k))

    return result.real


def _make_branch_level(level, k, D):
    """Create a checkpointed branch-level function with static level/k/D.

    jax.checkpoint traces all arguments, but level/k/D must stay as
    concrete Python ints (they control loop bounds and array shapes).
    Closing over them avoids this issue.
    """

    @jax.checkpoint
    def _branch_level(gammas, betas, F, log_scale):
        F_max = jnp.max(jnp.abs(F))
        F = jnp.where(F_max > 0, F / F_max, F)
        log_scale = log_scale + (D - 1) * jnp.log(jnp.where(F_max > 0, F_max, 1.0))
        child = pow_precise(F, D - 1)
        F = _hyperedge_branch(gammas, betas, level + 1, k, child_branch=child)
        return F, log_scale

    return _branch_level


def _contract_inner(gammas, betas, p, D, k):
    """Inner contraction function — pure JAX, suitable for jit + grad.

    Parameters
    ----------
    gammas, betas : jnp.array of shape (p,)
        QAOA angles.
    p, D, k : int
        Problem parameters (must be static for JIT).

    Returns
    -------
    scalar (float)
        <Z^{otimes k}>
    """
    if p == 0:
        return 0.0

    log_scale = jnp.float64(0.0)

    # Level 1: leaf
    F = _hyperedge_branch(gammas, betas, 1, k, child_branch=None)

    # Levels 2..p — each level is checkpointed so the backward pass
    # recomputes intermediates instead of storing all p levels' tensors
    # in GPU memory simultaneously.
    for level in range(1, p):
        branch_fn = _make_branch_level(level, k, D)
        F, log_scale = branch_fn(gammas, betas, F, log_scale)

    # Normalize before final (D-1) power
    F_max = jnp.max(jnp.abs(F))
    F = jnp.where(F_max > 0, F / F_max, F)
    log_scale = log_scale + (D - 1) * jnp.log(jnp.where(F_max > 0, F_max, 1.0))

    rb = pow_precise(F, D - 1)

    # Root contraction
    raw = _root_contract(rb, gammas, betas, p, D, k)

    # Apply accumulated scale
    raw = raw * jnp.exp(k * log_scale)
    return raw


# Module-level JIT-compiled functions.  Defined once so JAX's trace cache
# persists across calls — recompiles only when (p, D, k) changes, not on
# every optimizer iteration.
_contract_jitted = jax.jit(_contract_inner, static_argnums=(2, 3, 4))


@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5))
def _jvp_one(gammas, betas, p, D, k, param_idx):
    """Forward-mode JVP for a single parameter direction.

    param_idx 0..p-1  -> tangent in gamma[i]
    param_idx p..2p-1 -> tangent in beta[i-p]

    Memory: O(4^p) — just primal + one tangent. No backward graph.
    """
    g_dot = jnp.zeros_like(gammas)
    b_dot = jnp.zeros_like(betas)
    g_dot = g_dot.at[param_idx].set(1.0) if param_idx < p else g_dot
    b_dot = b_dot.at[param_idx - p].set(1.0) if param_idx >= p else b_dot

    primals, tangent_out = jax.jvp(
        lambda g, b: _contract_inner(g, b, p, D, k),
        (gammas, betas),
        (g_dot, b_dot),
    )
    return primals, tangent_out


# Track which (p, D, k) configs have been compiled to print status.
_compiled_configs = set()  # for contract_symmetric_tree
_compiled_grad_configs = set()  # for contract_with_grad


def contract_symmetric_tree(gammas, betas, p, D, k=2):
    """Exact <Z^{otimes k}> for depth-p QAOA on a D-regular k-uniform tree.

    Pure JAX implementation — all operations via jax.numpy, @jax.jit compiled.
    Float64 only (no DD support).

    Cost: O(p * 4^p), independent of D, k, and N_lc.

    Parameters
    ----------
    gammas, betas : array-like of shape (p,)
        QAOA angles.
    p : int
        QAOA depth.
    D : int
        Vertex degree.
    k : int
        Hyperedge size (default 2).

    Returns
    -------
    float
        The expectation value <Z^{otimes k}>.
    """
    import sys
    import time as _time

    gammas = jnp.asarray(gammas, dtype=jnp.float64)
    betas = jnp.asarray(betas, dtype=jnp.float64)

    key = (p, D, k)
    if key not in _compiled_configs:
        print(f"  [jax] compiling contract for p={p}, D={D}, k={k} " f"(4^p={4**p:,} entries)...", file=sys.stderr, flush=True)
        t0 = _time.perf_counter()
        result = float(_contract_jitted(gammas, betas, p, D, k))
        dt = _time.perf_counter() - t0
        _compiled_configs.add(key)
        print(f"  [jax] compiled in {dt:.1f}s", file=sys.stderr, flush=True)
        return result

    return float(_contract_jitted(gammas, betas, p, D, k))


def contract_with_grad(gammas, betas, p, D, k=2):
    """Compute <Z^{otimes k}> and its gradient via forward-mode JVP.

    Uses jax.jvp (forward-mode AD) with 2p sequential tangent directions.
    Each direction propagates through the forward pass alongside the primal,
    using O(4^p) memory — no backward graph, no XLA scheduling issues.

    Cost: 2p forward passes on GPU.
    Memory: O(4^p) per pass (~8 GiB at p=14, fits on any GPU).

    Parameters
    ----------
    gammas, betas : array-like of shape (p,)
        QAOA angles.
    p : int
        QAOA depth.
    D : int
        Vertex degree.
    k : int
        Hyperedge size (default 2).

    Returns
    -------
    value : float
        The expectation value <Z^{otimes k}>.
    grad_gammas : ndarray of shape (p,)
        Gradient w.r.t. gammas.
    grad_betas : ndarray of shape (p,)
        Gradient w.r.t. betas.
    """
    import sys
    import time as _time

    import numpy as np

    gammas = jnp.asarray(gammas, dtype=jnp.float64)
    betas = jnp.asarray(betas, dtype=jnp.float64)

    key = (p, D, k)
    if key not in _compiled_grad_configs:
        print(f"  [jax] compiling forward-mode JVP for p={p}, D={D}, k={k} " f"(2p={2*p} passes, 4^p={4**p:,} entries)...", file=sys.stderr, flush=True)
        t0 = _time.perf_counter()
        # Compile with first tangent direction
        val, _ = _jvp_one(gammas, betas, p, D, k, 0)
        dt = _time.perf_counter() - t0
        _compiled_grad_configs.add(key)
        print(f"  [jax] compiled in {dt:.1f}s", file=sys.stderr, flush=True)

    # Run 2p JVP passes: 0..p-1 for gammas, p..2p-1 for betas
    grad_g = np.empty(p)
    grad_b = np.empty(p)
    val = None
    for i in range(2 * p):
        v, t = _jvp_one(gammas, betas, p, D, k, i)
        if val is None:
            val = float(v)
        if i < p:
            grad_g[i] = float(t)
        else:
            grad_b[i - p] = float(t)

    return val, grad_g, grad_b


def light_cone_size(p, D, k):
    """Number of qubits in the depth-p light cone.

    N_lc = k * (a^{p+1} - 1) / (a - 1), where a = (D-1)(k-1).
    """
    a = (D - 1) * (k - 1)
    if a <= 0:
        return k
    if a == 1:
        return k * (p + 1)
    return k * (a ** (p + 1) - 1) // (a - 1)
