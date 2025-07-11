#!/usr/bin/env python
"""
Run a quick sweep over p-layers for portfolio QAOA and dump results to CSV.

• Uses the *analytic-gradient* path exposed by QOKit, so L-BFGS-B converges
  in far fewer calls than BOBYQA.
• Works entirely on CPU; later we’ll extend --device gpu and back-ends.

Example
-------
python scripts/run_sweep.py 12 4 0.7 --p 1 2 --optim lbfgs
"""
from __future__ import annotations
import argparse, csv, datetime, pathlib
import numpy as np

from qokit.portfolio_optimization import get_problem
from qokit.qaoa_objective_portfolio import get_qaoa_portfolio_objective


# --------------------------------------------------------------------------- #
#  optimisation helpers                                                       #
# --------------------------------------------------------------------------- #
def optimise_lbfgs(obj, x0):
    """SciPy L-BFGS-B with analytic gradients."""
    from scipy.optimize import minimize

    def fun(theta):
        val, grad = obj(theta, grad=True)
        return float(val.real), grad.astype(float)

    res = minimize(fun, x0, method="L-BFGS-B", jac=True,
                   options={"maxiter": 200})
    return res.x, res.nfev


def optimise_bobyqa(obj, x0):
    """Legacy NLopt-BOBYQA (gradient-free)."""
    from nlopt import opt, LN_BOBYQA

    o = opt(LN_BOBYQA, len(x0))
    o.set_min_objective(lambda x, g: obj(x).real)
    o.set_initial_step(0.2)
    o.set_maxeval(200)
    theta_opt = o.optimize(x0)
    return theta_opt, o.get_numevals()


# --------------------------------------------------------------------------- #
#  single sweep run                                                           #
# --------------------------------------------------------------------------- #
def run_once(N, K, q, p, optimiser="lbfgs"):
    po = get_problem(N=N, K=K, q=q, pre="rule")
    obj = get_qaoa_portfolio_objective(po, p=p, jac=True,
                                       precomputed_energies="vectorized")

    x0 = np.random.default_rng(0).random(2 * p)

    if optimiser == "lbfgs":
        theta_opt, n_calls = optimise_lbfgs(obj, x0)
    elif optimiser == "bobyqa":
        theta_opt, n_calls = optimise_bobyqa(obj, x0)
    else:
        raise ValueError("unknown optimiser")

    energy = obj(theta_opt).real
    return dict(p=p, energy=energy, f_calls=n_calls)


# --------------------------------------------------------------------------- #
#  CLI entry-point                                                            #
# --------------------------------------------------------------------------- #
def main() -> None:
    pa = argparse.ArgumentParser()
    pa.add_argument("N", type=int, help="asset universe size")
    pa.add_argument("K", type=int, help="cardinality constraint")
    pa.add_argument("q", type=float, help="risk-aversion parameter")
    pa.add_argument("--p", type=int, nargs="+", default=[1, 2, 3],
                    help="list of p layers to try")
    pa.add_argument("--optim", choices=["lbfgs", "bobyqa"], default="lbfgs",
                    help="choose optimiser (default: lbfgs)")
    args = pa.parse_args()

    rows = [run_once(args.N, args.K, args.q, p, optimiser=args.optim)
            for p in sorted(args.p)]

    # -------- save CSV -----------------------------------------------------
    out_dir = pathlib.Path("results"); out_dir.mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d")
    csv_path = out_dir / f"cpu_sweep_N{args.N}_K{args.K}_q{args.q}_{stamp}.csv"

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)

    # -------- pretty-print summary ----------------------------------------
    print(f"\np-layer sweep complete  →  {csv_path}")
    for r in rows:
        print(f"  p={r['p']:<2}  energy={r['energy']:+.6f}  "
              f"f_calls={r['f_calls']}")
    print()


if __name__ == "__main__":
    main()
