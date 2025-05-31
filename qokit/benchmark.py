from qokit.fur import get_available_simulator_names
from qokit.portfolio_optimization import get_problem

print(get_available_simulator_names("x"))

import numpy as np
from tqdm import tqdm
import networkx as nx
import timeit
from qokit.qaoa_objective_labs import get_qaoa_labs_objective
from qokit.qaoa_objective_maxcut import get_qaoa_maxcut_objective
from qokit.qaoa_objective import get_qaoa_objective

#the QAOA for portfolio optimization problem use the following functions :
# get the data for simulator : get_problem
# get the objective function : get_qaoa_portfolio_objective
#get the initial state : get_sk_ini

# number of qubits
for N in [12, 16]:
    print(f"N={N}")
    # QAOA depth
    p = 6

    theta = np.random.uniform(0,1,2*p)
    po_problem = get_problem(N=N, K=3, q=0.5, seed=1, pre='rule')


    # Function initialization may not be fast
    f_labs = get_qaoa_labs_objective(N, p)
    f_portfolio = get_qaoa_objective(N, p, objective="expectation", simulator="x")

    # Function evaluation is fast
    for f, label in [(f_labs, "LABS")]:
        f(theta) # do not count the first evaluation
        times = []
        for _ in tqdm(range(10)):
            start = timeit.default_timer()
            f(theta)
            end = timeit.default_timer()
            times.append(end-start)
        print(f"\t{label} finished in {np.mean(times):.4f} on average, min: {np.min(times):.4f}, max: {np.max(times):.4f}")


