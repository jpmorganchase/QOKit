import time
import matplotlib.pyplot as plt
from qokit.portfolio_optimization import get_problem_vectorized, get_problem, get_sk_ini
from qokit.qaoa_objective_portfolio import get_qaoa_portfolio_objective
from qokit.portfolio_optimization import get_configuration_cost_kw, po_obj_func, portfolio_brute_force
from qokit.utils import precompute_energies, reverse_array_index_bit_order
import cProfile, pstats
import multiprocessing
import numpy as np

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

if __name__ == '__main__':
    #profile_get_problem()
    profile_get_objective(N=22)
    #profile_init(p=6)




