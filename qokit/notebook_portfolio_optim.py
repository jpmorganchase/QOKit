import numpy as np
from qokit.portfolio_optimization import get_problem, get_problem_vectorized
from qokit.portfolio_optimization import portfolio_brute_force, get_sk_ini
from qokit.qaoa_objective_portfolio import get_qaoa_portfolio_objective
from scipy.optimize import minimize
import nlopt
import cProfile
import pstats
import qokit.config
import importlib

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
def main():
    #po_problem = get_problem(N=24, K=3, q=0.5, seed=1, pre=1)
    #means_in_spins = np.array([po_problem['means'][i] - po_problem['q'] * np.sum(po_problem['cov'][i, :]) for i in range(len(po_problem['means']))])
    #scale = 1 / np.sqrt(np.mean((( po_problem['q']*po_problem['cov'])**2).flatten())+np.mean((means_in_spins**2).flatten()))

    #po_problem = get_problem(N=24,K=3,q=0.5,seed=1,pre=scale)
    #po_problem2 = get_problem(N=24,K=3,q=0.5,seed=1,pre='rule')


    # confirm that the scaling rule in the function matches the one above
#    assert np.allclose(po_problem['cov'], po_problem2['cov'])
#    assert np.allclose(po_problem['means'], po_problem2['means'])

    p = 10
    qaoa_obj = get_qaoa_portfolio_objective(po_problem=po_problem2,p=p,ini='dicke',mixer='trotter_ring',T=1,simulator='python')
    best_portfolio = portfolio_brute_force(po_problem2,return_bitstring=False)

    #x0 = get_sk_ini(p=p)
    # Alternative: random initial point# x0 = np.random.rand(2*p)

    #po_energy = qaoa_obj(x0).real
    #po_ar = (po_energy-best_portfolio[1])/(best_portfolio[0]-best_portfolio[1])
    #print(f"energy = {po_energy}, Approximation ratio = {po_ar}")



    #_, opt_energy = minimize_nlopt(qaoa_obj, x0, p=1, rhobeg=0.01/1)
    #opt_ar = (opt_energy-best_portfolio[1])/(best_portfolio[0]-best_portfolio[1])
    #print(f"energy = {opt_energy}, Approximation ratio = {opt_ar}")

    #res = minimize(qaoa_obj, x0, method='COBYLA', options={'rhobeg':0.001})
    #print(f"energy = {res.fun}, Approximation ratio = {(res.fun-best_portfolio[1])/(best_portfolio[0]-best_portfolio[1])}")

if __name__ == "__main__":
    qokit.config.USE_NUMBA = "True"
    import qokit.fur.python.fur
    importlib.reload(qokit.fur.python.fur)
    import qokit.fur.python.qaoa_fur
    importlib.reload(qokit.fur.python.qaoa_fur)
    import qokit.fur.python.qaoa_simulator
    importlib.reload(qokit.fur.python.qaoa_fur)
    import qokit.fur.python.qaoa_simulator
    importlib.reload(qokit.fur.python.qaoa_simulator)
    import qokit.portfolio_optimization
    importlib.reload(qokit.portfolio_optimization)

    po_problem2 = get_problem(N=27, K=3, q=0.5, seed=1, pre='rule')
    profiler = cProfile.Profile()
    profiler.enable()
    p = 30
    qaoa_obj = get_qaoa_portfolio_objective(po_problem=po_problem2, p=p, ini='dicke', mixer='trotter_ring', T=1,
                                            simulator='python',precomputed_energies="vectorized")
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(20)