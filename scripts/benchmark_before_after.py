#!/usr/bin/env python
"""
Run both baseline and enhanced CPU profiles in one go and output
    results/before_after.csv           (merged)
    results/figure_before_after.png    (plot)

Profile switches:
  baseline → loop cost + BOBYQA     (slow)
  enhanced → vector + numba + lbfgs (fast)
"""
from __future__ import annotations
import argparse, itertools, csv, pathlib, time, os, numpy as np
import sys
import qokit.config as config_numba
import importlib

from qokit.qaoa_circuit_portfolio import generate_dicke_state_fast

# Add project root to sys.path
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qokit.portfolio_optimization import get_problem, brute_force_cost_vector
from qokit.qaoa_objective_portfolio import get_qaoa_portfolio_objective


# --------------------------------------------------------------------------- #
def make_baseline_obj(po, p):
    os.environ["QOKIT_DISABLE_VECTOR"] = "1"
    os.environ["QOKIT_DISABLE_CACHE"]  = "1"
    return get_qaoa_portfolio_objective(po, p=p, precomputed_energies=None)

def make_enhanced_obj(po, p):
    # vector + cache stay ON  (default behaviour)
    os.environ.pop("QOKIT_DISABLE_VECTOR", None)
    os.environ.pop("QOKIT_DISABLE_CACHE",  None)
    return get_qaoa_portfolio_objective(po, p=p,
                                        precomputed_energies="vectorized")


# --- optimiser shims ----------------------------------------------------- #
def run_bobyqa(obj, x0):
    import nlopt
    o = nlopt.opt(nlopt.LN_BOBYQA, len(x0))
    o.set_min_objective(lambda x, g: obj(x).real)
    o.set_maxeval(200); o.optimize(x0)

def run_lbfgs(obj, x0):
    from scipy.optimize import minimize
    minimize(obj, x0, method="L-BFGS-B", jac=False,
             options={"maxiter": 200})

# --- single timed run ---------------------------------------------------- #
def timed(fn):
    t0 = time.perf_counter(); fn(); return time.perf_counter() - t0

def one_case(N, p, q, Kfac, profile):
    po  = get_problem(N=N, K=int(Kfac*N), q=q, pre="rule")
    x0  = np.random.default_rng(0).random(2*p)
    #x0= generate_dicke_state_fast(N, int(Kfac*N))

    if profile == "baseline":
        qokit.config.USE_NUMBA = False
        import qokit.fur.python.fur
        importlib.reload(qokit.fur.python.fur)
        import qokit.fur.python.qaoa_fur
        importlib.reload(qokit.fur.python.qaoa_fur)
        import qokit.fur.python.qaoa_simulator
        importlib.reload(qokit.fur.python.qaoa_simulator)
        os.environ["QOKIT_DISABLE_VECTOR"] = "1"
        os.environ["QOKIT_DISABLE_CACHE"] = "1"
        obj = get_qaoa_portfolio_objective(po, p=p,ini="dicke", precomputed_energies=None,simulator="python")
        t = timed(lambda: run_bobyqa(obj, x0))
    else:
        os.environ.pop("QOKIT_DISABLE_VECTOR", None)
        os.environ.pop("QOKIT_DISABLE_CACHE", None)
        config_numba.USE_NUMBA = True
        import qokit.fur.python.fur
        importlib.reload(qokit.fur.python.fur)
        import qokit.fur.python.qaoa_fur
        importlib.reload(qokit.fur.python.qaoa_fur)
        import qokit.fur.python.qaoa_simulator
        importlib.reload(qokit.fur.python.qaoa_simulator)
        obj = get_qaoa_portfolio_objective(po, p=p, ini="dicke",precomputed_energies="vectorized",simulator="python")
        t = timed(lambda: run_lbfgs(obj, x0))
    return t

# --- CLI ----------------------------------------------------------------- #
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--Ns", nargs="+", type=int, default=[16])
    pa.add_argument("--ps", nargs="+", type=int, default=[1])
    pa.add_argument("--q", type=float,  default=0.7)
    pa.add_argument("--Kfac", type=float, default=0.3)
    pa.add_argument("--profile", choices=["baseline", "enhanced"],
                    default="enhanced")
    args = pa.parse_args()

    rows = [dict(N=N, p=p,
                 runtime=one_case(N,p,args.q,args.Kfac,args.profile),
                 profile=args.profile)
            for N,p in itertools.product(args.Ns, args.ps)]

    out = pathlib.Path("results");
    out.mkdir(exist_ok=True)
    csv_path = out / f"{args.profile}_cpu.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader();
        writer.writerows(rows)
    print("CSV →", csv_path)

    # merge & plot if both files exist
    base, enh = out/"baseline_cpu.csv", out/"enhanced_cpu.csv"
    if base.exists() and enh.exists():
        import pandas as pd, matplotlib.pyplot as plt
        b = pd.read_csv(base); e = pd.read_csv(enh)
        m = b.merge(e, on=["N","p"], suffixes=("_base","_enh"))
        m["percent_gain"] = 100*(m["runtime_base"]-m["runtime_enh"]) / m["runtime_base"]
        m.to_csv(out/"before_after.csv", index=False)
        for p, g in m.groupby("p"):
            plt.plot(g["N"], g["percent_gain"], marker="o", label=f"p={p}")
        plt.xlabel("Assets N"); plt.ylabel("% gain"); plt.legend()
        plt.tight_layout(); plt.savefig(out/"figure_before_after.png", dpi=200)
        print("Merged CSV + figure written.")
