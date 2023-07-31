###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import pytest
import typing
import numpy as np
from functools import partial
from qiskit import QuantumCircuit
from qiskit import Aer

from qokit.qaoa_objective_labs import get_precomputed_labs_merit_factors
from qokit.fur.lazy_import import MPI
from qokit.fur import QAOAFURXSimulatorGPUMPI, get_available_simulators
from qokit.fur import QAOAFURXSimulatorC, QAOAFURXSimulator, QAOAFURXSimulatorGPU, QAOAFastSimulatorBase
from qokit.labs import energy_vals_from_bitstring_general, get_energy_term_indices, get_terms_offset
from qokit.utils import precompute_energies
from qokit.qaoa_circuit_labs import get_qaoa_circuit

SIMULATORS = get_available_simulators("x")
print(SIMULATORS)


def _run_qiskit_mixer(N, betas):
    backend = Aer.get_backend("aer_simulator_statevector")
    qc = QuantumCircuit(N)
    for beta in betas:
        qc.rx(2 * beta, list(range(N)))
    qc.save_state()  # type: ignore
    return np.asarray(backend.run(qc).result().get_statevector())


def _check_simulator_against_qiskit(sim, N, terms, gammas, betas, sv0=None):
    res = sim.simulate_qaoa(gammas, betas, sv0=sv0)
    sv_qokit = sim.get_statevector(res)
    backend = Aer.get_backend("aer_simulator_statevector")
    qc = get_qaoa_circuit(N, terms, betas, gammas)
    sv_qiskit = np.asarray(backend.run(qc).result().get_statevector())
    assert np.allclose(sv_qiskit, sv_qokit), f"results from simulator {sim.__class__.__name__} do not match with qiskit"


@pytest.mark.parametrize("simclass", SIMULATORS)
@pytest.mark.parametrize("N, p", [(1, 1)])
def test_one_gate(N, p, simclass):
    """
    apply X on |0> state. This test allows to debug by looking at the actual SV values
    """
    gammas = [0] * p
    betas = [np.pi / 2] * p
    costs = np.zeros(2**N)
    sv0 = np.array([1] + [0] * (2**N - 1))

    sim = simclass(N, costs)
    res = sim.simulate_qaoa(gammas, betas, sv0=sv0)
    sv_qokit = sim.get_statevector(res)
    sv_qiskit = _run_qiskit_mixer(N, betas)
    assert np.allclose(sv_qiskit, sv_qokit), f"results from simulator {simclass.__name__} do not match with qiskit"


@pytest.mark.parametrize("simclass", SIMULATORS)
@pytest.mark.parametrize("N, p", [(5, 1), (5, 2)])
def test_mixer(N, p, simclass):
    """
    apply uniform Rx on |0> state
    """
    gammas = [0] * p
    betas = np.random.uniform(0, np.pi, p)
    costs = np.zeros(2**N)
    sv0 = np.array([1] + [0] * (2**N - 1))

    sim = simclass(N, costs)
    res = sim.simulate_qaoa(gammas, betas, sv0=sv0)
    sv_qokit = sim.get_statevector(res)
    sv_qiskit = _run_qiskit_mixer(N, betas)
    assert np.allclose(sv_qiskit, sv_qokit), f"results from simulator {simclass.__name__} do not match with qiskit"


@pytest.mark.parametrize("simclass", SIMULATORS)
@pytest.mark.parametrize("N, p", [(3, 1), (3, 2)])
def test_phase_operator(N, p, simclass):
    """
    apply phase operator on uniform superposition
    """
    gammas = np.random.uniform(0, np.pi, p)
    betas = [0] * p
    terms = [(3, [0, 1, 2]), (1, [0])]
    terms_without_weights = [t[1] for t in terms]
    offset = 2
    f = partial(energy_vals_from_bitstring_general, terms=terms_without_weights, offset=offset)
    precomputed_energies = precompute_energies(f, N) - offset

    sim = simclass(N, costs=precomputed_energies)
    _check_simulator_against_qiskit(sim, N, terms_without_weights, gammas, betas)


@pytest.mark.parametrize("simclass", SIMULATORS)
@pytest.mark.parametrize("N, p", [(3, 1), (3, 2)])
def test_phase_operator_labs(N, p, simclass):
    """
    apply phase operator on uniform superposition (using cost from LABS)
    """
    gammas = np.random.uniform(0, np.pi, p)
    betas = [0] * p
    terms_ix, offset = get_energy_term_indices(N)
    precomputed_energies = -(N**2) / (2 * get_precomputed_labs_merit_factors(N)) - offset

    sim = simclass(N, costs=precomputed_energies)
    _check_simulator_against_qiskit(sim, N, terms_ix, gammas, betas)


@pytest.mark.parametrize("simclass", SIMULATORS)
def test_against_qiskit_p_1(simclass):
    beta = np.random.uniform(0, np.pi)
    gamma = np.random.uniform(0, np.pi)

    N = 5
    terms_ix, offset = get_energy_term_indices(N)
    precomputed_energies = -(N**2) / (2 * get_precomputed_labs_merit_factors(N)) - offset

    sim = simclass(N, costs=precomputed_energies)
    _check_simulator_against_qiskit(sim, N, terms_ix, [gamma], [beta])


@pytest.mark.parametrize("simclass", SIMULATORS)
def test_against_qiskit_p_3(simclass):
    p = 3
    N = 5
    terms, offset = get_energy_term_indices(N)
    precomputed_energies = -(N**2) / (2 * get_precomputed_labs_merit_factors(N)) - offset

    sim = simclass(N, costs=precomputed_energies)
    for _ in range(5):
        betas = np.random.uniform(0, np.pi, p)
        gammas = np.random.uniform(0, np.pi, p)

        _check_simulator_against_qiskit(sim, N, terms, gammas, betas)


@pytest.mark.parametrize("simclass", SIMULATORS)
@pytest.mark.parametrize("N", [4, 5])
def test_terms_api(simclass: typing.Type[QAOAFastSimulatorBase], N):
    """Test that simulator can precompute energies when given wegihted Terms"""
    p = 1
    terms, offset = get_terms_offset(N)
    terms_ix, _ = get_energy_term_indices(N)

    sim = simclass(N, terms=terms)
    precomputed_energies = -(N**2) / (2 * get_precomputed_labs_merit_factors(N)) - offset
    energies = sim.get_cost_diagonal()
    assert np.allclose(energies, precomputed_energies), f"results from simulator {sim.__class__.__name__} do not match with precomputed energies"
    for _ in range(5):
        betas = np.random.uniform(0, np.pi, p)
        gammas = np.random.uniform(0, np.pi, p)

        _check_simulator_against_qiskit(sim, N, terms_ix, gammas, betas)
