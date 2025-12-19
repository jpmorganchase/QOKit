import networkx as nx
from qokit.maxcut import get_maxcut_terms
from qokit.parameter_utils import get_fixed_gamma_beta
import qokit
import numpy as np
from qokit.qaoa_circuit_maxcut import get_ws_qaoa_circuit
from qokit.fur.diagonal_precomputation import precompute_vectorized_cpu_parallel
from qiskit_aer import AerSimulator


def test_furxz_backends():

    N = 10
    d = 3
    p = 1
    seed = 1
    G = nx.random_regular_graph(d, N, seed=seed)
    terms = get_maxcut_terms(G)
    gamma, beta = get_fixed_gamma_beta(d, p)
    ini_rots = np.random.rand(N)

    simclass = qokit.fur.choose_simulator_xz(name="c")
    sim = simclass(N, terms=terms)
    _result = sim.simulate_ws_qaoa(gamma, beta, ini_rots)
    c_energy = sim.get_expectation(_result)

    simclass = qokit.fur.choose_simulator_xz(name="python")
    sim = simclass(N, terms=terms)
    _result = sim.simulate_ws_qaoa(gamma, beta, ini_rots)
    python_energy = sim.get_expectation(_result)

    assert np.isclose(c_energy, python_energy)


def test_ws_degeneracy():

    N = 10
    d = 3
    p = 1
    seed = 1
    G = nx.random_regular_graph(d, N, seed=seed)
    terms = get_maxcut_terms(G)
    precomputed_objectives = precompute_vectorized_cpu_parallel(terms, 0.0, N)
    gamma, beta = get_fixed_gamma_beta(d, p)
    ini_rots = np.pi / 2 * np.ones(N)

    simclass = qokit.fur.choose_simulator_xz(name="python")
    sim = simclass(N, terms=terms)
    _result = sim.simulate_ws_qaoa(gamma, beta, ini_rots)
    ws_energy = sim.get_expectation(_result)

    simclass = qokit.fur.choose_simulator(name="python")
    sim = simclass(N, terms=terms)
    _result = sim.simulate_qaoa(gamma, beta)
    qaoa_energy = sim.get_expectation(_result)

    qc = get_ws_qaoa_circuit(G, gamma, beta, ini_rots)
    backend = AerSimulator(method="statevector")
    sv = np.asarray(backend.run(qc).result().get_statevector())
    qiskit_prob = np.abs(sv) ** 2
    qiskit_energy = np.dot(qiskit_prob, precomputed_objectives)

    assert np.isclose(ws_energy, qaoa_energy)
    assert np.isclose(qiskit_energy, qaoa_energy)


def test_qiskit_qokit():
    ##### qiskit circuit

    N = 10
    d = 3
    p = 1
    seed = 1
    G = nx.random_regular_graph(d, N, seed=seed)
    terms = get_maxcut_terms(G)
    gamma, beta = get_fixed_gamma_beta(d=d, p=p)
    ini_rots = np.random.rand(N)

    simclass = qokit.fur.choose_simulator_xz(name="c")
    sim = simclass(N, terms=terms)
    _result = sim.simulate_ws_qaoa(list(np.asarray(gamma)), list(np.asarray(beta)), ini_rots)
    qaoa_prob = sim.get_probabilities(_result)

    qc = get_ws_qaoa_circuit(G, gamma, beta, ini_rots)
    backend = AerSimulator(method="statevector")
    sv = np.asarray(backend.run(qc).result().get_statevector())
    qiskit_prob = np.abs(sv) ** 2

    assert np.allclose(qaoa_prob, qiskit_prob)
