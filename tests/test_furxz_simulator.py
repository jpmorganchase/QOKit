import networkx as nx
from qokit.maxcut import get_maxcut_terms
from qokit.parameter_utils import get_fixed_gamma_beta
import qokit
import numpy as np

def test_furxz_backends():

    N = 10
    d = 3
    p = 1
    seed = 1
    G = nx.random_regular_graph(d,N,seed=seed)
    terms = get_maxcut_terms(G)
    gamma, beta = get_fixed_gamma_beta(d,p)
    ini_rots = np.random.rand(N)

    simclass = qokit.fur.choose_simulator_xz(name='c')
    sim = simclass(N, terms=terms)
    _result = sim.simulate_ws_qaoa(gamma, beta, ini_rots)
    c_energy = sim.get_expectation(_result)
    
    simclass = qokit.fur.choose_simulator_xz(name='python')
    sim = simclass(N, terms=terms)
    _result = sim.simulate_ws_qaoa(gamma, beta, ini_rots)
    python_energy = sim.get_expectation(_result)
    
    assert np.isclose(c_energy, python_energy)