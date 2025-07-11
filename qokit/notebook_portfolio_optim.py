# run_po_qaoa.py  — python run_po_qaoa.py 12 4 0.7
import argparse, numpy as np
from qokit.portfolio_optimization import get_problem, get_sk_ini, portfolio_brute_force
from qokit.qaoa_objective_portfolio import get_qaoa_portfolio_objective
from nlopt import opt, LN_BOBYQA          # pip install nlopt

def main(N, K, q, p=2, layers=1, mixer='trotter_ring'):
    po = get_problem(N=N, K=K, q=q, pre='rule')      # auto-scale rule
    qaoa_obj = get_qaoa_portfolio_objective(po, p=p, ini='dicke',
                                            mixer=mixer, T=layers,
                                            simulator='cupy' )  # GPU if you have it

    best_energy, *_ = portfolio_brute_force(po)
    x0 = get_sk_ini(p)                               # SK angles, already rescaled
    o = opt(LN_BOBYQA, 2*p)
    o.set_min_objective(lambda x, g: qaoa_obj(x).real)
    o.set_initial_step(0.2);  o.set_maxeval(200)
    xopt = o.optimize(x0)
    final = qaoa_obj(xopt).real
    print(f"best classical: {best_energy:.4f}  |  QAOA: {final:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('N', type=int);  parser.add_argument('K', type=int)
    parser.add_argument('q', type=float)
    args = parser.parse_args()
    main(args.N, args.K, args.q)
