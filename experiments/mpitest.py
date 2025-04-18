import numpy as np
import networkx as nx
from qokit.parameter_utils import get_fixed_gamma_beta
from qokit.qaoa_objective_maxcut import get_qaoa_maxcut_objective
from qokit.maxcut import get_maxcut_terms

N = 10
d = 3
seed = 1
p = 4
G = nx.random_regular_graph(d, N, seed=seed)
terms = get_maxcut_terms(G)
f_python = get_qaoa_maxcut_objective(N, p, G, simulator="gpu", parameterization="gamma beta")
f_gpu = get_qaoa_maxcut_objective(N, p, G, simulator='gpu', parameterization='gamma beta')
#f_gpumpi = get_qaoa_maxcut_objective(N, p, G, simulator="gpumpi", parameterization="gamma beta")
gamma, beta = get_fixed_gamma_beta(d, p)
print(f"python energy: {f_python(gamma, beta)}")
print(f'gpu energy: {f_gpu(gamma, beta)}')
#print(f"gpumpi energy: {f_gpumpi(gamma, beta)}")
