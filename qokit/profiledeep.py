import time
import matplotlib.pyplot as plt
from qokit.portfolio_optimization import get_problem_vectorized, get_problem, get_sk_ini
from qokit.qaoa_objective_portfolio import get_qaoa_portfolio_objective
from qokit.portfolio_optimization import get_configuration_cost_kw, po_obj_func, portfolio_brute_force
from qokit.utils import precompute_energies, reverse_array_index_bit_order
import cProfile, pstats
import multiprocessing
import numpy as np
import nlopt
import importlib

def profile_get_problem(N=22, pre='rule'):
    #first call to call get data and avoid connection issue
    K, q, seed = 3, 0.5, 1
    po_problem = get_problem(N=N, K=K, q=q, seed=seed, pre=pre)

    profiler = cProfile.Profile()
    profiler.enable()
    po_problem1 = get_problem(N=N, K=K, q=q, seed=seed, pre=pre)
    profiler.disable()
    stats = pstats.Stats(profiler)
    print(f"Cumulative stats length: {len(stats.stats)}")
    stats.sort_stats('cumulative').print_stats(20)

    profiler = cProfile.Profile()
    profiler.enable()
    po_problem2 = get_problem_vectorized(N=N, K=K, q=q, seed=seed, pre=pre)
    profiler.disable()
    stats = pstats.Stats(profiler)
    print(f"Cumulative stats length: {len(stats.stats)}")
    stats.sort_stats('cumulative').print_stats(20)



def profile_get_objective(N = 22,p = 6,simulator = "python"):
    """

    """
    K, q, seed, pre = 3, 0.5, 1, 'rule'
    po_problem = get_problem_vectorized(N=N, K=K, q=q, seed=seed, pre=pre)

    profiler = cProfile.Profile()
    profiler.enable()
    qaoa = get_qaoa_portfolio_objective(po_problem=po_problem, p=p, ini='dicke', mixer='trotter_ring', T=1,
                                        simulator=simulator)
    profiler.disable()
    stats = pstats.Stats(profiler)
    print(f"Cumulative stats length: {len(stats.stats)}")
    stats.sort_stats('cumulative').print_stats(20)

    profiler = cProfile.Profile()
    profiler.enable()
    qaoa = get_qaoa_portfolio_objective(po_problem=po_problem, p=p, ini='dicke', mixer='trotter_ring', T=1,
                                        simulator=simulator, precomputed_energies="vectorized")
    profiler.disable()
    stats = pstats.Stats(profiler)
    print(f"Cumulative stats length: {len(stats.stats)}")
    stats.sort_stats('cumulative').print_stats(20)

def profile_init(p=6):
    profiler = cProfile.Profile()
    profiler.enable()
    x0 = get_sk_ini(p=p)
    print(x0)
    profiler.disable()
    stats = pstats.Stats(profiler)
    print(f"Cumulative stats length: {len(stats.stats)}")
    stats.sort_stats('cumulative').print_stats(20)

def minimize_nlopt(f, x0, rhobeg=None, p=None):
    def nlopt_wrapper(x, grad):
        if grad.size > 0:
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

if __name__ == '__main__':
    #profile_get_problem()
    #profile_get_objective(N=22)
    #profile_init(p=6)
    import qokit.fur.python.fur

    importlib.reload(qokit.fur.python.fur)
    import qokit.fur.python.qaoa_fur

    importlib.reload(qokit.fur.python.qaoa_fur)
    import qokit.fur.python.qaoa_simulator

    importlib.reload(qokit.fur.python.qaoa_simulator)

    N= 23
    p=1
    K, q, seed, pre = 3, 0.5, 1, 'rule'
    po_problem = get_problem(N=N, K=K, q=q, seed=seed, pre=pre)

    profiler = cProfile.Profile()
    profiler.enable()
    po_problem = get_problem(N=N, K=K, q=q, seed=seed, pre=pre)
    po_obj = po_obj_func(po_problem)
    qaoa_obj = get_qaoa_portfolio_objective(po_problem=po_problem, p=p, ini='dicke', mixer='trotter_ring', T=1,
                                            simulator='python', precomputed_energies=None)
    #x0 = get_sk_ini(p=p)

    #po_energy = qaoa_obj(x0).real
    #_, opt_energy = minimize_nlopt(qaoa_obj, x0, p=p, rhobeg=0.01 / 1)
    profiler.disable()
    stats = pstats.Stats(profiler)
    print(f"Cumulative stats length: {len(stats.stats)}")
    stats.sort_stats('cumulative').print_stats(50)



