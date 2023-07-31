###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import pytest
import numpy as np
from qiskit import QuantumCircuit, execute
import qiskit.circuit.library
from qiskit import Aer

from qokit.fur import QAOAFURXYRingSimulator, QAOAFURXYRingSimulatorC, QAOAFURXYRingSimulatorGPU
from qokit.fur import QAOAFURXYCompleteSimulator, QAOAFURXYCompleteSimulatorC, QAOAFURXYCompleteSimulatorGPU
from qokit.fur import get_available_simulators

SIMULATORS = get_available_simulators("xyring")


def _create_rxy_circuit_qiskit(N, index_pairs, betas: list):
    """
    Appliy qiskit.circuit.library.XXPlusYYGate to the circuit, at the given
    indices in corresponding order of index pairs.
    Indices may not be unique, which allows to apply several equivalent layers
    of gates for one element of betas list
    """
    qc = QuantumCircuit(N)
    qc.rx(np.pi, 0)
    for beta in betas:
        for i, j in index_pairs:
            qc.append(qiskit.circuit.library.XXPlusYYGate(beta * 2), [i, j])
    qc.save_state()  # type: ignore
    return qc


def _check_simulator_against_qiskit(sim, N, index_pairs, gammas, betas, sv0=None, n_trotters=1):
    _r = sim.simulate_qaoa(gammas, betas, sv0=sv0, n_trotters=n_trotters)
    res = sim.get_statevector(_r)
    backend = Aer.get_backend("aer_simulator_statevector")
    betas_qiskit = [b / n_trotters for b in betas for _ in range(n_trotters)]
    qc = _create_rxy_circuit_qiskit(N, index_pairs, betas_qiskit)
    sv_qiskit = execute(qc, backend).result().get_statevector()
    assert sv_qiskit.equiv(res)


@pytest.mark.parametrize("simclass", SIMULATORS)
@pytest.mark.parametrize("N, p, n_trotters", [(4, 1, 1), (5, 2, 1), (4, 1, 2), (5, 1, 2), (5, 2, 2)])
def test_rxy_ring_trotter(N, p, n_trotters, simclass):
    """
    Test QAOA with gamma=0, using ring topology and initial state |0...01>
    n_trotters = 1 corresponds to simple rxy_ring application
    """
    index_pairs_ring = [(i, i + 1) for i in range(0, N - 1, 2)] + [(i, i + 1) for i in range(1, N - 1, 2)]
    index_pairs_ring += [(0, N - 1)]
    init_state = np.zeros(2**N)
    init_state[1] = 1
    gammas = np.zeros(p)
    betas = np.random.uniform(0, np.pi, p)

    sim = simclass(N, costs=np.zeros(2**N))
    _check_simulator_against_qiskit(sim, N, index_pairs_ring, gammas, betas, init_state, n_trotters=n_trotters)


SIMULATORS = get_available_simulators("xycomplete")


@pytest.mark.parametrize("simclass", SIMULATORS)
@pytest.mark.parametrize("N, p, n_trotters", [(4, 1, 1), (5, 2, 1), (4, 1, 2), (5, 1, 2), (5, 2, 2)])
def test_rxy_complete_trotter(N, p, n_trotters, simclass):
    """
    Test QAOA with gamma=0, using ring topology and initial state |0...01>
    n_trotters = 1 corresponds to simple rxy_ring application
    """
    index_pairs_ring = [(i, j) for i in range(N - 1) for j in range(i + 1, N)]
    init_state = np.zeros(2**N)
    init_state[1] = 1
    gammas = np.zeros(p)
    betas = np.random.uniform(0, np.pi, p)

    sim = simclass(N, costs=np.zeros(2**N))
    _check_simulator_against_qiskit(sim, N, index_pairs_ring, gammas, betas, init_state, n_trotters=n_trotters)
