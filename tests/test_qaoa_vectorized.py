###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import numpy as np
from functools import partial
import qiskit
from qiskit import QuantumCircuit, execute
from qiskit.providers.aer import AerSimulator
import pytest

from qokit.qaoa_objective_labs import get_precomputed_labs_merit_factors

from qokit.labs import energy_vals_from_bitstring_general, get_energy_term_indices
from qokit.utils import precompute_energies
from qokit.qaoa_circuit_labs import get_qaoa_circuit
from qokit.qaoa_vectorized import QAOAvectorizedBackendSimulator, get_qaoa_statevector


def test_normalization():
    N = 4
    sim = QAOAvectorizedBackendSimulator(N, np.zeros(2**N))
    x = sim.wavefn.flatten()
    assert np.isclose(x.dot(x), 1)


def test_hadamard():
    N = 5
    sim = QAOAvectorizedBackendSimulator(N, np.zeros(2**N))
    for idx in range(N):
        sim.apply_hadamard(idx)
    assert np.isclose(sim.wavefn.flatten()[0], 1)


def test_rx():
    N = 5
    beta = np.random.uniform(0, np.pi)
    sim = QAOAvectorizedBackendSimulator(N, np.zeros(2**N))
    for idx in range(N):
        sim.apply_hadamard(idx)
    sim.apply_rx(3, beta)
    sv_vectorized = sim.wavefn.flatten()

    backend = AerSimulator(method="statevector")
    qc = QuantumCircuit(N)
    qc.rx(beta, 3)
    qc.save_state()
    sv_qiskit = np.asarray(backend.run(qc).result().get_statevector())
    assert np.allclose(sv_vectorized, sv_qiskit)

    backend = AerSimulator(method="statevector")
    qc = QuantumCircuit(N)
    qc.h(3)
    qc.rz(beta, 3)
    qc.h(3)
    qc.save_state()
    sv_qiskit = np.asarray(backend.run(qc).result().get_statevector())
    assert np.allclose(sv_vectorized, sv_qiskit)


def test_rxy():
    N = 2
    beta = np.random.uniform(0, np.pi)
    sim = QAOAvectorizedBackendSimulator(N, np.zeros(2**N))
    # note that QAOAvectorizedBackendSimulator starts with uniform superposition
    # have to undo it first
    for idx in range(N):
        sim.apply_hadamard(idx)
    sim.apply_rx(0, np.pi)
    sim.apply_rxy(0, 1, beta)
    sv_vectorized = sim.wavefn.flatten()

    qc = QuantumCircuit(N)
    qc.rx(np.pi, 0)
    qc.append(
        qiskit.circuit.library.XXPlusYYGate(beta),
        [0, 1],
    )
    qc.save_state()
    backend = AerSimulator(method="statevector")
    sv_qiskit = execute(qc, backend).result().get_statevector()
    assert np.allclose(sv_vectorized, sv_qiskit)


def test_rxy_multiple():
    N = 4
    indices_to_apply_xy = [(0, 1), (1, 2), (2, 3)]
    beta = np.random.uniform(0, np.pi)
    sim = QAOAvectorizedBackendSimulator(N, np.zeros(2**N))
    # note that QAOAvectorizedBackendSimulator starts with uniform superposition
    # have to undo it first
    for idx in range(N):
        sim.apply_hadamard(idx)
    sim.apply_rx(0, np.pi)
    for q1, q2 in indices_to_apply_xy:
        sim.apply_rxy(q1, q2, beta)
    sv_vectorized = sim.wavefn.flatten()

    qc = QuantumCircuit(N)
    qc.rx(np.pi, 0)

    for q1, q2 in indices_to_apply_xy:
        qc.append(
            qiskit.circuit.library.XXPlusYYGate(beta),
            [q1, q2],
        )
    qc.save_state()
    backend = AerSimulator(method="statevector")
    sv_qiskit = execute(qc, backend).result().get_statevector()
    assert np.allclose(sv_vectorized, sv_qiskit)


def test_apply_diagonal():
    N = 3
    gamma = np.random.uniform(0, np.pi)
    terms = [[0, 1, 2], [0]]
    offset = 2
    f = partial(energy_vals_from_bitstring_general, terms=terms, offset=offset)
    precomputed_energies = precompute_energies(f, N) - offset

    sim = QAOAvectorizedBackendSimulator(N, precomputed_energies)
    sim.apply_diagonal(gamma)
    sv_vectorized = sim.wavefn.flatten()

    backend = AerSimulator(method="statevector")
    qc = get_qaoa_circuit(N, terms, [0], [gamma])
    sv_qiskit = np.asarray(backend.run(qc).result().get_statevector())
    assert np.allclose(sv_vectorized, sv_qiskit)


def test_apply_diagonal_labs():
    N = 5
    gamma = np.random.uniform(0, np.pi)
    terms, offset = get_energy_term_indices(N)
    precomputed_energies = -(N**2) / (2 * get_precomputed_labs_merit_factors(N)) - offset

    sim = QAOAvectorizedBackendSimulator(N, precomputed_energies)
    sim.apply_diagonal(gamma)
    sv_vectorized = sim.wavefn.flatten()

    backend = AerSimulator(method="statevector")
    qc = get_qaoa_circuit(N, terms, [0], [gamma])
    sv_qiskit = np.asarray(backend.run(qc).result().get_statevector())

    assert np.allclose(sv_vectorized, sv_qiskit)


def test_against_qiskit_p_1():
    beta = np.random.uniform(0, np.pi)
    gamma = np.random.uniform(0, np.pi)

    N = 5
    terms, offset = get_energy_term_indices(N)
    precomputed_energies = -(N**2) / (2 * get_precomputed_labs_merit_factors(N)) - offset

    # vectorized
    sim = QAOAvectorizedBackendSimulator(N, precomputed_energies)
    sim.apply_diagonal(gamma)
    for idx in range(N):
        sim.apply_rx(idx, 2 * beta)
    sv_vectorized = sim.wavefn.flatten()

    # qiskit
    backend = AerSimulator(method="statevector")
    qc = get_qaoa_circuit(N, terms, [beta], [gamma])
    sv_qiskit = np.asarray(backend.run(qc).result().get_statevector())

    assert np.allclose(sv_vectorized, sv_qiskit)


def test_against_qiskit_p_3():
    p = 3
    N = 5
    terms, offset = get_energy_term_indices(N)
    precomputed_energies = -(N**2) / (2 * get_precomputed_labs_merit_factors(N)) - offset

    backend = AerSimulator(method="statevector")
    for _ in range(5):
        beta = np.random.uniform(0, np.pi, p)
        gamma = np.random.uniform(0, np.pi, p)

        # vectorized
        sim = QAOAvectorizedBackendSimulator(N, precomputed_energies)
        sim.apply_qaoa_circuit(beta, gamma)
        sv_vectorized = sim.wavefn.flatten()

        # qiskit
        qc = get_qaoa_circuit(N, terms, beta, gamma)
        sv_qiskit = np.asarray(backend.run(qc).result().get_statevector())

        assert np.allclose(sv_vectorized, sv_qiskit)


def test_get_qaoa_statevector():
    N = 6
    p = 3

    backend = AerSimulator(method="statevector")
    for _ in range(5):
        beta = np.random.uniform(0, np.pi, p)
        gamma = np.random.uniform(0, np.pi, p)

        terms, offset = get_energy_term_indices(N)
        precomputed_energies = -(N**2) / (2 * get_precomputed_labs_merit_factors(N)) - offset
        sv_vectorized = get_qaoa_statevector(beta, gamma, N=N, precomputed_energies=precomputed_energies)

        qc = get_qaoa_circuit(N, terms, beta, gamma)
        sv_qiskit = np.asarray(backend.run(qc).result().get_statevector())

        assert np.allclose(sv_vectorized, sv_qiskit)
