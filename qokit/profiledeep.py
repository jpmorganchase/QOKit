import time
import matplotlib.pyplot as plt
from qokit.portfolio_optimization import get_problem_vectorized
from qokit.qaoa_objective_portfolio import get_qaoa_portfolio_objective
from qokit.portfolio_optimization import get_configuration_cost_kw, po_obj_func, portfolio_brute_force
from qokit.utils import precompute_energies, reverse_array_index_bit_order
import cProfile, pstats
import multiprocessing
import numpy as np

if __name__ == '__main__':






    N = 22
    p = 6
    simulator = "python"
    K, q, seed, pre = 3, 0.5, 1, 'rule'
    po_problem = get_problem_vectorized(N=N, K=K, q=q, seed=seed, pre=pre)
    #po_obj = po_obj_func(po_problem)

    profiler = cProfile.Profile()
    profiler.enable()
    qaoa = get_qaoa_portfolio_objective(po_problem=po_problem, p=p, ini='dicke', mixer='trotter_ring', T=1, simulator=simulator)
    profiler.disable()
    stats = pstats.Stats(profiler)
    print(f"Cumulative stats length: {len(stats.stats)}")
    stats.sort_stats('cumulative').print_stats(20)

    profiler = cProfile.Profile()
    profiler.enable()
    qaoa = get_qaoa_portfolio_objective(po_problem=po_problem, p=p, ini='dicke', mixer='trotter_ring', T=1,
                                        simulator=simulator,precomputed_energies="vectorized")
    profiler.disable()
    stats = pstats.Stats(profiler)
    print(f"Cumulative stats length: {len(stats.stats)}")
    stats.sort_stats('cumulative').print_stats(20)
