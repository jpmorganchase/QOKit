import numpy as np
import time
from qokit.fur.python.qaoa_simulator import QAOAFURXYRingSimulator, QAOAFURXYRingVectorizedSimulator

def test_furxy_vectorized_vs_classical_functional():
    n_qubits = 4
    p = 2
    gammas = np.random.uniform(0, np.pi, p)
    betas = np.random.uniform(0, np.pi, p)
    costs = np.random.uniform(-1, 1, 2 ** n_qubits)

    sim_classical = QAOAFURXYRingSimulator(n_qubits, costs=costs)
    sim_vectorized = QAOAFURXYRingVectorizedSimulator(n_qubits, costs=costs)

    sv_classical = sim_classical.simulate_qaoa(gammas, betas)
    sv_vectorized = sim_vectorized.simulate_qaoa(gammas, betas)

    # Compare probabilities (statevector global phase is irrelevant)
    probs_classical = np.abs(sv_classical) ** 2
    probs_vectorized = np.abs(sv_vectorized) ** 2

    max_diff = np.max(np.abs(probs_classical - probs_vectorized))
    print("Max probability difference:", max_diff)
    print("Index of max difference:", np.argmax(np.abs(probs_classical - probs_vectorized)))
    print("Classical probs:", probs_classical)
    print("Vectorized probs:", probs_vectorized)

    np.testing.assert_allclose(probs_classical, probs_vectorized, atol=1e-10)

def test_furxy_vectorized_vs_classical_profile():
    n_qubits = 8
    p = 4
    gammas = np.random.uniform(0, np.pi, p)
    betas = np.random.uniform(0, np.pi, p)
    costs = np.random.uniform(-1, 1, 2 ** n_qubits)

    sim_classical = QAOAFURXYRingSimulator(n_qubits, costs=costs)
    sim_vectorized = QAOAFURXYRingVectorizedSimulator(n_qubits, costs=costs)

    start = time.perf_counter()
    sim_classical.simulate_qaoa(gammas, betas)
    time_classical = time.perf_counter() - start

    start = time.perf_counter()
    sim_vectorized.simulate_qaoa(gammas, betas)
    time_vectorized = time.perf_counter() - start

    print(f"Classical time: {time_classical:.6f}s, Vectorized time: {time_vectorized:.6f}s")

def test_furxy_ring_vs_vectorized_single_step():
    n_qubits = 4
    theta = np.random.uniform(0, np.pi)
    sv0 = np.random.randn(2**n_qubits) + 1j * np.random.randn(2**n_qubits)
    sv0 /= np.linalg.norm(sv0)

    from qokit.fur.python.fur import furxy_ring, furxy_ring_vectorized_single

    sv_classical = sv0.copy()
    sv_vectorized = sv0.copy()

    furxy_ring(sv_classical, theta, n_qubits)
    furxy_ring_vectorized_single(sv_vectorized, theta, n_qubits)

    diff = np.max(np.abs(sv_classical - sv_vectorized))
    print("Max statevector difference after one ring step:", diff)
    print("Classical statevector:", sv_classical)
    print("Vectorized statevector:", sv_vectorized)
    np.testing.assert_allclose(sv_classical, sv_vectorized, atol=1e-10)

