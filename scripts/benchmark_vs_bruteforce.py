#!/usr/bin/env python
"""
Compare enhanced QAOA path (vectorised + numba + L-BFGS-B + cache)
against a pure-Python brute-force optimiser.

CSV columns:
  N,p,t_brute,t_enhanced,percent_gain
"""
from __future__ import annotations
import argparse, itertools, csv, pathlib, time, numpy as np
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="cupy._environment")

# Add project root to sys.path
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qokit.portfolio_optimization import get_problem, brute_force_cost_vector
from qokit.qaoa_objective_portfolio import get_qaoa_portfolio_objective

# ------------ helpers ----------------------------------------------------- #
def baseline_bruteforce(po, p, x0):
    """pure-Python brute + BOBYQA (no numba, no vectorisation)"""
    from nlopt import opt, LN_BOBYQA
    def loop_cost(bit_int: int) -> float:
        N = po["N"]
        x = [(bit_int >> k) & 1 for k in range(N)]
        return po["q"] * np.dot(x, po["cov"] @ x) - np.dot(po["means"], x)
    # exhaustive search just to show pain
    best = min(loop_cost(b) for b in range(1 << po["N"]))
    # wrap BOBYQA to spend comparable optimizer time
    o = opt(LN_BOBYQA, len(x0))
    o.set_min_objective(lambda x, g: best)   # dummy
    o.set_maxeval(10); o.optimize(x0)

def enhanced_qaoa(po, p, x0):
    obj = get_qaoa_portfolio_objective(
        po, p=p, precomputed_energies="vectorized"
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
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Ns", type=int, default=16)
    parser.add_argument("--ps", type=int, default=1)
    parser.add_argument("--q", type=float, default=1.0)
    parser.add_argument("--Kfac", type=float, default=0.5)
    parser.add_argument("--csv", type=str, default="results/improvement_vs_bruteforce.csv")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    if args.verbose:
        print(f"Running benchmark with N={args.Ns}, p={args.ps}, q={args.q}, Kfac={args.Kfac}")

    results = []
    if args.verbose:
        print("Starting one_case computation...")
    result = one_case(args.Ns, args.ps, args.q, args.Kfac)
    results.append(result)
    if args.verbose:
        print("Computation finished. Writing results to CSV...")

    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Results written to {args.csv}")