import importlib
import matplotlib.pyplot as plt
import time
import qokit.config as qokit_config
from qokit.portfolio_optimization import get_problem, get_sk_ini
from qokit.qaoa_objective_portfolio import get_qaoa_portfolio_objective
import nlopt

def minimize_nlopt(f, x0, rhobeg=None, p=None):
    def nlopt_wrapper(x, grad):
        if grad.size > 0:
            import sys
            sys.exit("Shouldn't be calling a gradient!")
        return f(x).real
    opt = nlopt.opt(nlopt.LN_BOBYQA, 2 * p)
    opt.set_min_objective(nlopt_wrapper)
    opt.set_xtol_rel(1e-8)
    opt.set_ftol_rel(1e-8)
    opt.set_initial_step(rhobeg)
    xstar = opt.optimize(x0)
    minf = opt.last_optimum_value()
    return xstar, minf

def run_benchmark(use_numba,N=20):
    qokit_config.USE_NUMBA = use_numba
    import qokit.fur.python.fur
    importlib.reload(qokit.fur.python.fur)
    import qokit.fur.python.qaoa_fur
    importlib.reload(qokit.fur.python.qaoa_fur)
    import qokit.fur.python.qaoa_simulator
    importlib.reload(qokit.fur.python.qaoa_simulator)
    po_problem = get_problem(N=N, K=3, q=0.5, seed=1, pre="rule")
    p = 1
    qaoa_obj = get_qaoa_portfolio_objective(po_problem=po_problem, p=p, ini='dicke', mixer='trotter_ring', T=1, simulator='python', precomputed_energies="vectorized")
    x0 = get_sk_ini(p=p)
    start = time.perf_counter()
    po_energy = qaoa_obj(x0).real
    _, opt_energy = minimize_nlopt(qaoa_obj, x0, p=1, rhobeg=0.01/1)
    elapsed = time.perf_counter() - start
    return elapsed

timings = []
labels = []
for use_numba in [False, True]:
    label = "With Numba" if use_numba else "Without Numba"
    print(f"Running benchmark: {label}")
    elapsed = run_benchmark(use_numba, N=22)
    timings.append(elapsed)
    labels.append(label)
    print(f"{label}: {elapsed:.4f} seconds")

plt.bar(labels, timings)
plt.ylabel("Elapsed time (s)")
plt.title("QAOA Portfolio Optimization: Numba Speedup")
plt.show()
