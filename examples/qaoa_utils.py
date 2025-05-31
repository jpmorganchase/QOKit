# qaoa_utils.py

import numpy as np
import os
import time
import nlopt

from qiskit_aer import AerSimulator
from qokit.portfolio_optimization import get_sk_ini
from qokit.qaoa_objective_portfolio import get_qaoa_portfolio_objective


def minimize_nlopt(f, x0, rhobeg=None, p=None):
    """
    Helper function to run NLopt's BOBYQA optimizer.
    f: Objective function to minimize
    x0: Initial guess for parameters
    rhobeg: Initial step size (optional)
    p: QAOA depth (used to determine parameter count)
    """
    def nlopt_wrapper(x, grad): # grad is ignored by BOBYQA
        if grad.size > 0:
            pass 
        return f(x).real # Ensure scalar real value is returned

    opt = nlopt.opt(nlopt.LN_BOBYQA, 2 * p)
    opt.set_min_objective(nlopt_wrapper)

    opt.set_xtol_rel(1e-8) # Relative tolerance on parameters
    opt.set_ftol_rel(1e-8) # Relative tolerance on objective function value
    
    if rhobeg is not None:
        opt.set_initial_step(rhobeg)
    
    xstar = opt.optimize(x0)
    minf = opt.last_optimum_value()

    return xstar, minf


def run_single_optimization(run_config, po_problem_arg, best_portfolio_arg):
    """
    Performs a single QAOA optimization run with a given configuration.
    This function is designed to be executed by each parallel process.
    It now explicitly accepts 'po_problem_arg' and 'best_portfolio_arg'.
    """
    seed = run_config['seed']
    current_p = run_config['p'] 

    # Extract N from po_problem_arg as it's needed for AR calculation logic
    N = po_problem_arg['N'] 

    simulator_backend_for_process = AerSimulator(method='statevector')
    simulator_backend_for_process.set_options(max_parallel_threads=0, statevector_parallel_threshold=16)
    
    qaoa_obj_for_run = get_qaoa_portfolio_objective(
        po_problem=po_problem_arg, # <<< Now uses the passed argument
        p=current_p,
        ini='dicke',
        mixer='trotter_ring',
        T=1,
        simulator=simulator_backend_for_process,
        mixer_topology='linear'
    )

    np.random.seed(seed) 
    x0 = get_sk_ini(p=current_p)

    print(f"Starting optimization run with seed {seed} (process {os.getpid()})... ", end='')
    
    start_time = time.perf_counter()
    try:
        optimized_params, opt_energy = minimize_nlopt(qaoa_obj_for_run, x0, p=current_p, rhobeg=0.01 / current_p)
    except Exception as e:
        print(f"Optimization with seed {seed} failed: {e}")
        return None
    end_time = time.perf_counter()
    print(f"Completed in {end_time - start_time:.2f} seconds.")

    # --- Handling Approximation Ratio (AR) ---
    opt_ar_value = 'N/A (Brute-force not feasible or not computed)'
    # Use passed best_portfolio_arg and N from po_problem_arg
    if best_portfolio_arg[0] is not None and best_portfolio_arg[1] is not None and N <= 20:
        opt_ar_value = (opt_energy - best_portfolio_arg[1]) / (best_portfolio_arg[0] - best_portfolio_arg[1])
        print(f"Run {seed} (process {os.getpid()}): Energy = {opt_energy:.8f}, AR = {opt_ar_value:.8f}")
    else:
        print(f"Run {seed} (process {os.getpid()}): Energy = {opt_energy:.8f}")
    
    return {'seed': seed, 'energy': opt_energy, 'ar': opt_ar_value, 'optimized_params': optimized_params}