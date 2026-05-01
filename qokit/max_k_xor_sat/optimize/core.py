###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
"""Single-stage QAOA angle optimizer (BOBYQA or L-BFGS-B).

Optimizes QAOA angles (gammas, betas) for a given (k, D, p) by minimizing
<Z^k> (equivalently maximizing the energy (1 - <Z^k>)/2) via symmetric
tensor contraction.

Supports two optimizers:
  - BOBYQA (gradient-free, default): uses pybobyqa
  - L-BFGS-B (gradient-based): uses scipy with analytic gradients

When n_cheb is set, BOBYQA optimizes over Chebyshev coefficient space
(2*n_cheb parameters) instead of raw angle space (2p parameters). This
reduces BOBYQA's model-building cost from O(p) to O(sqrt(p)) evaluations —
critical for DD precision where each evaluation is expensive.
"""

import sys
import time

import numpy as _np

from qokit.max_k_xor_sat.optimize.seed import (
    angles_to_cheb,
    cheb_to_angles,
    load_seed_angles,
    save_output_file,
)


# ── Contraction function access ─────────────────────────────────
# Import from the package level so that callers can patch
# qokit.max_k_xor_sat.optimize.contract_symmetric_tree for an alternative backend.


def _get_contract_fn():
    """Get the contraction function (patchable via qokit.max_k_xor_sat.optimize)."""
    import qokit.max_k_xor_sat.optimize as _pkg

    return _pkg.contract_symmetric_tree


def _get_grad_fn():
    """Lazy import of the gradient function (patchable via qokit.max_k_xor_sat.optimize)."""
    import qokit.max_k_xor_sat.optimize as _pkg

    if _pkg._contract_with_grad is None:
        from qokit.max_k_xor_sat.contract import contract_with_grad

        _pkg._contract_with_grad = contract_with_grad
    return _pkg._contract_with_grad


# ── Objective function ──────────────────────────────────────────


def _make_objective(eval_fn, state, k, verbose, label="", track_best_x=False, on_improvement=None):
    """Build a tracked objective function for the optimizer."""

    def objective(x):
        val = float(eval_fn(x))
        state["n_evals"] += 1
        if abs(val) > 1.0:
            state["n_invalid"] += 1
            if verbose:
                elapsed = time.perf_counter() - state["t0"]
                print(
                    f"  eval {state['n_evals']:>4d}{label}: " f"UNSTABLE <Z^{k}> = {val:>.6e}  ({elapsed:.1f}s)",
                    file=sys.stderr,
                )
            return 1.0
        improved = val < state["best"]
        if val <= state["best"]:
            state["best"] = val
            if track_best_x:
                state["best_valid_x"] = _np.array(x, copy=True)
        if improved and on_improvement is not None:
            on_improvement(x, val)
        if verbose:
            elapsed = time.perf_counter() - state["t0"]
            energy = (1 - val) / 2
            best_energy = (1 - state["best"]) / 2
            print(
                f"  eval {state['n_evals']:>4d}{label}: " f"(1-<Z^{k}>)/2 = {energy:>.10f}" f"  best = {best_energy:>.10f}  ({elapsed:.1f}s)",
                file=sys.stderr,
            )
        return val

    return objective


# ── BOBYQA utilities ────────────────────────────────────────────


def _npt(n, maxfun):
    """Number of interpolation points for BOBYQA, clamped to budget."""
    default = 2 * n + 1
    minimum = n + 2
    if maxfun is not None and maxfun <= default:
        return min(max(minimum, maxfun - 1), maxfun - 1)
    return default


def _run_bobyqa(objective, x0, maxfun, rhobeg=None):
    """Run a single BOBYQA optimization pass."""
    import pybobyqa

    n = len(x0)
    if maxfun <= n + 2:
        return x0.copy()
    if rhobeg is None:
        rhobeg = min(0.5, max(0.05, 0.3 * float(_np.max(_np.abs(x0)))))
    soln = pybobyqa.solve(
        objective,
        x0,
        rhobeg=rhobeg,
        rhoend=1e-6,
        maxfun=maxfun,
        npt=_npt(n, maxfun),
        do_logging=False,
        print_progress=False,
    )
    return soln.x


# ── L-BFGS-B optimizer ─────────────────────────────────────────


def _run_lbfgs(gammas, betas, p, k, D, maxiter, verbose, state, precision="float64", on_improvement=None):
    """L-BFGS-B optimization over full 2p angle space using analytic gradients."""
    from scipy.optimize import minimize

    grad_fn = _get_grad_fn()
    x0 = _np.concatenate([gammas, betas])

    val0, dg0, db0 = grad_fn(gammas, betas, p, D, k, precision=precision)
    val0 = float(val0)
    g0 = _np.concatenate([dg0, db0])
    state["n_evals"] += 1
    if abs(val0) <= 1.0 and val0 < state["best"]:
        state["best"] = val0
        state["best_valid_x"] = x0.copy()
        if on_improvement is not None:
            on_improvement(x0, val0)
    gnorm = float(_np.max(_np.abs(g0)))
    h = min(0.03, max(0.001, 0.01 / max(gnorm, 1e-10)))

    def value_and_grad(z):
        x = x0 + h * z
        g, b = x[:p], x[p:]
        val, dg, db = grad_fn(g, b, p, D, k, precision=precision)
        val = float(val)
        grad_x = _np.concatenate([dg, db])
        grad_z = h * grad_x
        state["n_evals"] += 1

        if abs(val) > 1.0:
            state["n_invalid"] += 1
            if verbose:
                elapsed = time.perf_counter() - state["t0"]
                print(
                    f"  eval {state['n_evals']:>4d} [lbfgs]: " f"UNSTABLE <Z^{k}> = {val:>.6e}  ({elapsed:.1f}s)",
                    file=sys.stderr,
                )
            return 1.0, _np.zeros_like(grad_z)

        improved = val < state["best"]
        if val <= state["best"]:
            state["best"] = val
            state["best_valid_x"] = x.copy()
        if improved and on_improvement is not None:
            on_improvement(x, val)

        if verbose:
            elapsed = time.perf_counter() - state["t0"]
            energy = (1 - val) / 2
            best_energy = (1 - state["best"]) / 2
            print(
                f"  eval {state['n_evals']:>4d} [lbfgs]: " f"(1-<Z^{k}>)/2 = {energy:>.10f}" f"  best = {best_energy:>.10f}  ({elapsed:.1f}s)",
                file=sys.stderr,
            )

        return val, grad_z

    a = (D - 1) * (k - 1)
    noise = max(a ** (p / 2.0) * 1e-16, 1e-14)
    ftol = max(noise * 10, 1e-10)
    gtol = h * max(noise * 100, 1e-5)

    z0 = _np.zeros(2 * p)
    minimize(
        value_and_grad,
        z0,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": maxiter,
            "maxfun": maxiter * 3,
            "ftol": ftol,
            "gtol": gtol,
        },
    )


# ── Steepest descent with FD gradient ────────────────────────────


def _run_steepest(gammas, betas, p, k, D, maxiter, verbose, state, precision="float64", on_improvement=None, n_cheb=None):
    """Single steepest-descent step: FD gradient + line search.

    Computes the gradient via sequential central finite differences
    in the full 2p angle space (one eval at a time — no extra memory
    beyond a single forward pass). Then does a golden-section line
    search along -grad.

    Memory: O(4^p) per eval (forward only, no adjoint cache).
    Cost: 2*(2p)+1 evals for gradient + ~10 evals for line search.
    """
    contract_fn = _get_contract_fn()
    x0 = _np.concatenate([gammas, betas])
    n = len(x0)

    # Central value (reuse probe if available)
    if state["best"] <= 1.0:
        f0 = state["best"]
    else:
        f0 = float(contract_fn(gammas, betas, p, D, k, precision=precision))
        state["n_evals"] += 1

    # FD gradient: sequential central differences, one perturbation at a time
    a = (D - 1) * (k - 1)
    noise = max(a ** (p / 2.0) * 2.3e-16, 1e-14) if a > 1 else 1e-14
    h = max(min(noise ** (1.0 / 3.0), 1e-2), 1e-8)

    grad = _np.zeros(n)
    if verbose:
        print(f"  --- FD gradient ({n} params, h={h:.1e}) ---", file=sys.stderr)

    for i in range(n):
        x_plus = x0.copy()
        x_plus[i] += h
        f_plus = float(contract_fn(x_plus[:p], x_plus[p:], p, D, k, precision=precision))
        state["n_evals"] += 1

        x_minus = x0.copy()
        x_minus[i] -= h
        f_minus = float(contract_fn(x_minus[:p], x_minus[p:], p, D, k, precision=precision))
        state["n_evals"] += 1

        grad[i] = (f_plus - f_minus) / (2 * h)

    gnorm = float(_np.linalg.norm(grad))
    direction = -grad / max(gnorm, 1e-12)

    if verbose:
        elapsed = time.perf_counter() - state["t0"]
        print(f"  --- gradient computed: |g|={gnorm:.6e}  ({elapsed:.1f}s) ---", file=sys.stderr)

    if gnorm < 1e-12:
        return  # already at optimum

    # Golden-section line search along x0 + alpha * direction
    def eval_alpha(alpha):
        x = x0 + alpha * direction
        val = float(contract_fn(x[:p], x[p:], p, D, k, precision=precision))
        state["n_evals"] += 1
        if abs(val) <= 1.0 and val < state["best"]:
            state["best"] = val
            state["best_valid_x"] = x.copy()
            if on_improvement is not None:
                on_improvement(x, val)
        if verbose:
            elapsed = time.perf_counter() - state["t0"]
            energy = (1 - val) / 2
            best_energy = (1 - state["best"]) / 2
            print(
                f"  eval {state['n_evals']:>4d} [line a={alpha:.4e}]: "
                f"(1-<Z^{k}>)/2 = {energy:>.10f}"
                f"  best = {best_energy:>.10f}  ({elapsed:.1f}s)",
                file=sys.stderr,
            )
        return val

    # Find bracket: start small, double until function stops improving
    alpha = min(0.01, h * 100)
    best_alpha, best_f = 0.0, f0
    for _ in range(10):
        f_cur = eval_alpha(alpha)
        if state["n_evals"] >= maxiter:
            break
        if f_cur < best_f:
            best_alpha = alpha
            best_f = f_cur
            alpha *= 2.0
        else:
            break

    # Golden section search within [0, alpha]
    gr = (1 + 5**0.5) / 2
    a_lo, a_hi = 0.0, alpha
    for _ in range(min(maxiter - state["n_evals"], 6)):
        if a_hi - a_lo < alpha * 0.01:
            break
        a1 = a_hi - (a_hi - a_lo) / gr
        a2 = a_lo + (a_hi - a_lo) / gr
        f1 = eval_alpha(a1)
        if state["n_evals"] >= maxiter:
            break
        f2 = eval_alpha(a2)
        if state["n_evals"] >= maxiter:
            break
        if f1 < f2:
            a_hi = a2
        else:
            a_lo = a1


# ── Main entry point ────────────────────────────────────────────


def optimize_angles(k, D, p, maxiter=200, n_cheb=None, output_file=None, verbose=False, precision="float64", optimizer="bobyqa", seed_fn=None):
    """Optimize QAOA angles to maximize (1 - <Z^k>)/2.

    Single-stage optimization using either BOBYQA (gradient-free),
    L-BFGS-B (gradient-based with analytic gradients), or steepest
    descent (FD gradient + line search, forward-only memory).

    Parameters
    ----------
    k : int
        Hyperedge size (uniformity).
    D : int
        Vertex degree (regularity).
    p : int
        QAOA depth.
    maxiter : int
        Maximum function evaluations (BOBYQA) or iterations (L-BFGS-B).
    n_cheb : int, optional
        Number of Chebyshev coefficients per angle group for BOBYQA.
    output_file : str, optional
        Path to output file for seeding / saving.
    verbose : bool
        Print per-evaluation diagnostics to stderr.
    precision : str
        'float64' (default) or 'dd' (double-double, ~32 digits).
    optimizer : str
        'bobyqa' (gradient-free, default), 'lbfgs' (gradient-based
        via reverse-mode adjoint), or 'steepest' (single FD gradient
        + line search, forward-only memory — for high-p DD where the
        adjoint doesn't fit).
    seed_fn : callable, optional
        Custom seed loader: ``(k, D, p) -> (gammas, betas, source_str)``.

    Returns
    -------
    dict
        Keys: gammas, betas, expectation (<Z^k>), objective ((1-<Z^k>)/2),
        num_evals, converged, seed_source.
    """
    contract_fn = _get_contract_fn()

    # Load seed
    if seed_fn is not None:
        gammas0, betas0, seed_source = seed_fn(k, D, p)
    else:
        gammas0, betas0, seed_source = load_seed_angles(k, D, p, output_file)

    state = {
        "n_evals": 0,
        "best": _np.inf,
        "best_valid_x": None,
        "n_invalid": 0,
        "t0": time.perf_counter(),
    }

    # Checkpoint callback
    def _checkpoint_angles(x, val):
        if output_file is None:
            return
        entry = {
            "gammas": _np.asarray(x[:p]).tolist(),
            "betas": _np.asarray(x[p:]).tolist(),
            "expectation": float(val),
            "objective": float((1 - val) / 2),
            "num_evals": state["n_evals"],
            "converged": True,
            "seed_source": seed_source,
        }
        try:
            save_output_file(output_file, k, D, {str(p): entry})
        except Exception:
            pass

    # Probe seed quality
    val_probe = float(contract_fn(gammas0, betas0, p, D, k, precision=precision))
    state["n_evals"] += 1
    if abs(val_probe) <= 1.0:
        state["best"] = val_probe
        state["best_valid_x"] = _np.concatenate([gammas0, betas0]).copy()
        _checkpoint_angles(state["best_valid_x"], val_probe)
    if verbose:
        elapsed = time.perf_counter() - state["t0"]
        energy = (1 - val_probe) / 2
        print(
            f"  eval    1 [probe]: " f"(1-<Z^{k}>)/2 = {energy:>.10f}  ({elapsed:.1f}s)",
            file=sys.stderr,
        )

    if verbose:
        names = {"lbfgs": "L-BFGS-B", "bobyqa": "BOBYQA", "steepest": "Steepest descent"}
        method_name = names.get(optimizer, optimizer)
        use_cheb_bobyqa = n_cheb is not None and p > 1 and optimizer == "bobyqa"
        n_c = min(n_cheb, p) if use_cheb_bobyqa else None
        paramstr = f"{2*n_c} cheb params" if n_c else f"{2*p} params"
        print(f"  Method: {method_name} ({paramstr})", file=sys.stderr)

    # ── L-BFGS-B path ──
    if optimizer == "lbfgs":
        _run_lbfgs(
            gammas0,
            betas0,
            p,
            k,
            D,
            maxiter,
            verbose,
            state,
            precision=precision,
            on_improvement=_checkpoint_angles,
        )

    # ── Steepest descent path (FD gradient + line search) ──
    elif optimizer == "steepest":
        _run_steepest(
            gammas0,
            betas0,
            p,
            k,
            D,
            maxiter,
            verbose,
            state,
            precision=precision,
            on_improvement=_checkpoint_angles,
            n_cheb=n_cheb,
        )

    # ── BOBYQA path ──
    else:
        use_cheb = n_cheb is not None and p > 1
        n_c = min(n_cheb, p) if use_cheb else None

        if use_cheb:
            c_gamma = angles_to_cheb(gammas0, n_c)
            c_beta = angles_to_cheb(betas0, n_c)
            x0_cheb = _np.concatenate([c_gamma, c_beta])

            def eval_cheb(x):
                g = cheb_to_angles(x[:n_c], p)
                b = cheb_to_angles(x[n_c:], p)
                return contract_fn(g, b, p, D, k, precision=precision)

            def cheb_on_improvement(x_cheb, val):
                g = cheb_to_angles(x_cheb[:n_c], p)
                b = cheb_to_angles(x_cheb[n_c:], p)
                x_angles = _np.concatenate([g, b])
                state["best_valid_x"] = x_angles
                _checkpoint_angles(x_angles, val)

            objective = _make_objective(
                eval_cheb,
                state,
                k,
                verbose,
                label=" [cheb]",
                track_best_x=False,
                on_improvement=cheb_on_improvement,
            )

            rhobeg = min(0.1, max(0.01, 0.05 * float(_np.max(_np.abs(x0_cheb)))))
            _run_bobyqa(objective, x0_cheb, maxiter, rhobeg=rhobeg)

        else:
            x0 = _np.concatenate([gammas0, betas0])

            def eval_main(x):
                return contract_fn(x[:p], x[p:], p, D, k, precision=precision)

            objective = _make_objective(
                eval_main,
                state,
                k,
                verbose,
                track_best_x=True,
                on_improvement=_checkpoint_angles,
            )
            _run_bobyqa(objective, x0, maxiter)

    # Extract best result
    if state["best_valid_x"] is not None:
        best_x = state["best_valid_x"]
    else:
        best_x = _np.concatenate([gammas0, betas0])

    opt_gammas = best_x[:p].copy()
    opt_betas = best_x[p:].copy()

    if state["best"] <= 1.0:
        expectation = state["best"]
    else:
        expectation = float(contract_fn(opt_gammas, opt_betas, p, D, k, precision=precision))

    if state["n_invalid"] > 0:
        print(
            f"  warning: {state['n_invalid']}/{state['n_evals']} evaluations " f"hit numerical instability (|<Z^{k}>| > 1)",
            file=sys.stderr,
        )

    return {
        "gammas": opt_gammas.tolist(),
        "betas": opt_betas.tolist(),
        "expectation": float(expectation),
        "objective": float((1 - expectation) / 2),
        "num_evals": state["n_evals"],
        "converged": state["best"] <= 1.0,
        "seed_source": seed_source,
    }
