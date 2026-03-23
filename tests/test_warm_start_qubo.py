import numpy as np
import networkx as nx
import random
from qokit.warm_start import WSSolver, WSSolverQUBO, maxcut_qubo_from_G, get_terms_from_QUBO
from qokit.fur import get_available_simulator_names
import qokit
from qokit.maxcut import maxcut_obj, get_adjacency_matrix 
from qokit.fur.diagonal_precomputation import precompute_vectorized_cpu_parallel
from qokit.qaoa_circuit import get_ws_qaoa_circuit_from_terms
from qiskit_aer import Aer
import pytest

simulators_to_run = get_available_simulator_names("xz") 

def test_qubo_matrix():
    N = 10
    G = nx.random_regular_graph(3, N, seed=1)
    Q0 = maxcut_qubo_from_G(G)
    Q = np.zeros((N,N))
    nodes = list(G.nodes())
    for u,v,data in G.edges(data=True):
        w = data.get('weight', 1.0)

        Q[u,u] += w
        Q[v,v] += w
        Q[u,v] += -1 * w
        Q[v,u] += -1 * w
    assert np.allclose(Q0,Q)
    
def test_maxcut_qubo():
    N = 10
    G = nx.random_regular_graph(3, N, seed=1)
    Q = maxcut_qubo_from_G(G)
    x = [random.choice([0, 1]) for _ in range(N)]
    assert np.isclose(x @ Q @ x, maxcut_obj(np.asarray(x), get_adjacency_matrix(G)))     
    
def test_obj_in_p_space():
    # p_i = sin**2(theta_i/2), sin**2(theta_i) = 4p(1-p)
    # f(p) = \sum_i Q_ii p_i + 2 \sum_{i<j} Q_ij p_i p_j - \lamda \sum_i 4p_i(1-p_i)
    # let B = Q - D (without diagonal)
    # f'(p) = diag(Q) - 4 \lambda 1 + 2 B p + 8 \lambda p
    # f'(p_i) = Q_ii + 2 \sum_{j!=i} Q_ij pj - 4 \lambda + 8 \lambda p_i = 0 ==> p^star_i = (-Q_ii - 2 \sum_{j!=i} Q_ij pj + 4 \lambda) / (8 * \lambda)
    # f''(p) = 2 B + 8 \lambda I

    N = 10
    G = nx.random_regular_graph(3, N, seed=1)
    ws_solver = WSSolver(G)
    Q = -maxcut_qubo_from_G(G)
    
    theta = np.random.rand(N)
    lamd = 0.35
    p = np.sin(theta/2)**2
    exp_H, var_H, gE, gV = ws_solver.get_p0_std_quantities(theta)
    exp_H2 = ws_solver.get_p0_cut(theta)
    assert np.isclose(exp_H, exp_H2)
    
    obj1 = exp_H + 4 * lamd * np.sum( p * (1-p))
    
    obj2 = ws_solver.rws_objective(theta, lamd=lamd)
    
    qubo_obj = 0 
    qubo_obj += np.inner(np.diag(Q), p)
    for i in range(N):
        for j in range(N):
            if i!=j:
                qubo_obj += Q[i,j]*p[i]*p[j]
    assert not np.isclose(p.T@Q@p, qubo_obj) ##### NOT USE p.T@Q@p
    
    obj3 = -1 * (qubo_obj - 4 * lamd * np.sum( p * (1-p)))
    assert np.isclose(obj1, obj2)
    assert np.isclose(obj1, obj3)
    
@pytest.mark.parametrize("simulator", simulators_to_run)
def test_po_qubo(simulator):
    from qokit.portfolio_optimization import get_problem, po_obj_func
    from qokit.qaoa_objective import get_qaoa_objective
    from qokit.utils import precompute_energies, reverse_array_index_bit_order

    N = 14
    seed = 1
    po_problem = get_problem(N=N,K=3,q=0.5,seed=seed,pre='rule')
    po_obj = po_obj_func(po_problem)
    precomputed_energies = reverse_array_index_bit_order(precompute_energies(po_obj, N)).real
    

    p = 1
    gamma, beta = np.random.rand(p), np.random.rand(p)
    
        
    qaoa_obj = get_qaoa_objective(N=N, precomputed_diagonal_hamiltonian=po_problem["scale"] * precomputed_energies,mixer='x',simulator=simulator)
    x0 = np.concatenate((gamma,beta))
    po_energy = qaoa_obj(x0).real/po_problem["scale"]

    cov = po_problem['cov'].copy()
    Q = (po_problem['q'] * cov - np.diag( po_problem['means'])) / po_problem['scale']
    qubo_ws_solver = WSSolverQUBO(Q)
    energy_from_Q = qubo_ws_solver.run_standard_qaoa(po_problem['scale']*gamma, beta) ## TODO: this needs to match!
    assert np.isclose(energy_from_Q, po_energy)

    Q = (po_problem['q'] * cov - np.diag( po_problem['means'])) / po_problem['scale']
    x = [random.choice([0, 1]) for _ in range(N)]
    assert np.isclose(x @ Q @ x, po_obj(np.asarray(x)))

def test_qubo_mean_and_variance():
    N = 10
    G = nx.random_regular_graph(3, N, seed=1)
    ws_solver = WSSolver(G)
    theta = np.random.rand(N)
    e, var, ge, gvar = ws_solver.get_p0_std_quantities(theta)
    
    Q = maxcut_qubo_from_G(G)
    qubo_ws_solver = WSSolverQUBO(Q)
    e2, var2, ge2, gvar2 = qubo_ws_solver.product_state_stats_and_grads_from_Q(theta)
    
    assert np.isclose(e,e2)
    assert np.isclose(var,var2)
    assert np.allclose(ge,ge2)
    assert np.allclose(gvar,gvar2)
    
def test_qubo_qiskit_circuit():
    N = 10
    Q = np.random.rand(N,N)
    terms = get_terms_from_QUBO(Q)
    precomputed_objectives = precompute_vectorized_cpu_parallel(terms, 0.0, N)

    p = 2
    gamma, beta = np.random.rand(p), np.random.rand(p)
    theta = np.random.rand(N)
    
    circ = get_ws_qaoa_circuit_from_terms(N, terms[:-1], gamma, beta, theta, save_statevector=True)
    backend = Aer.get_backend("aer_simulator_statevector")
    sv = np.asarray(backend.run(circ).result().get_statevector())
    qiskit_prob = np.abs(sv) ** 2
    qiskit_energy = np.dot(qiskit_prob, precomputed_objectives)

    simclass = qokit.fur.choose_simulator_xz(name='auto')
    sim = simclass(N, terms=terms)
    cost = sim.get_cost_diagonal()
    best_cut = np.max(cost)
    _result = sim.simulate_ws_qaoa(list(np.asarray(gamma)),
                                    list(np.asarray(beta)),
                                    np.asarray(theta))
    qokit_energy = sim.get_expectation(_result)
    qokit_prob = sim.get_probabilities(_result)

    assert np.allclose(qiskit_prob,qokit_prob)
    assert np.isclose(qiskit_energy,qokit_energy)
