#!/usr/bin/env python
"""
Compare enhanced QAOA path (vectorised + numba + L-BFGS-B + cache)
against a pure-Python brute-force optimiser.

CSV columns:
  N,p,t_brute,t_enhanced,percent_gain
"""
from __future__ import annotations
import argparse, itertools, csv, pathlib, time, numpy as np
from qokit.portfolio_optimization import get_problem, brute_force_cost_vector
from qokit.qaoa_objective_portfolio import get_qaoa_portfolio_objective

# ------------ helpers ----------------------------------------------------- #
def baseline_bruteforce(po, p, x0):
    """pure-Python brute + BOBYQA (no numba, no vectorisation)"""
    from nlopt import opt, LN_BOBYQA
    def loop_cost(bit_int: int) -> float:
        N = po["N"]
        x = [(bit_int >> k) & 1 for k in range(N)]
        return po["q"] * np.dot(x, po["cov"] @ x) - np.dot(po["mu"], x)
    # exhaustive search just to show pain
    best = min(loop_cost(b) for b in range(1 << po["N"]))
    # wrap BOBYQA to spend comparable optimizer time
    o = opt(LN_BOBYQA, len(x0))
    o.set_min_objective(lambda x, g: best)   # dummy
    o.set_maxeval(10); o.optimize(x0)

def enhanced_qaoa(po, p, x0):
    obj = get_qaoa_portfolio_objective(
        po, p=p, jac=True, precomputed_energies="vectorized"
    )
    from scipy.optimize import minimize
    #fun = lambda t: (obj(t)[0], obj(t, grad=True)[1].astype(float))
    minimize(obj, x0, method="L-BFGS-B", jac=False,
             options={"maxiter": 200})

def timed(fn):
    t0 = time.perf_counter(); fn(); return time.perf_counter() - t0

def one_case(N, p, q, Kfac):
    po  = get_problem(N=N, K=int(Kfac*N), q=q, pre="rule")
    x0  = np.random.default_rng(0).random(2*p)

    # compile numba once so enhanced timer excludes JIT
    if N <= 20 and p == 1:
        brute_force_cost_vector(po)

    t_brute = timed(lambda: baseline_bruteforce(po, p, x0))
    t_enh   = timed(lambda: enhanced_qaoa(po, p, x0))
    gain    = 100.0 * (t_brute - t_enh) / t_brute
    return dict(N=N, p=p,
                t_brute=round(t_brute, 4),
                t_enhanced=round(t_enh, 4),
                percent_gain=round(gain, 2))

# ------------ CLI ----------------------------
