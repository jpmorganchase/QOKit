###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import warnings
import pytest
import numpy as np
from functools import partial
from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
import numba.cuda

from qokit.qaoa_objective_labs import get_precomputed_labs_merit_factors
from qokit.fur import QAOAFURXSimulatorC, QAOAFURXSimulator, QAOAFURXSimulatorGPU
from qokit.labs import energy_vals_from_bitstring_general, get_energy_term_indices
from qokit.utils import precompute_energies
from qokit.qaoa_circuit_labs import get_qaoa_circuit


def _get_sim_method(sim, idx):
    return getattr(sim, f"simulate_qaoa_{idx}")


def _run_all_simulators(N, costs, gammas, betas, sv0=None):
    res = {}

    betas = np.asarray(betas) * 2

    for simclass in (QAOAFURXSimulator,):
        sim = simclass(N, costs)
        res[simclass.__name__] = sim.simulate_qaoa(gammas, betas, sv0=sv0)

    for simclass in (QAOAFURXSimulatorC,):
        sim = simclass(N, costs)
        res[simclass.__name__] = sim.simulate_qaoa(gammas, betas, sv0=sv0).get_complex()

    if numba.cuda.is_available():
        for simclass in (QAOAFURXSimulatorGPU,):
            sim = simclass(N, costs)
            res[simclass.__name__] = sim.simulate_qaoa(gammas, betas, sv0=sv0)
    else:
        warnings.warn("Skip GPU tests as no compatible devices are found.")

    return res


def _run_qiskit(N, terms, gammas, betas):
    backend = AerSimulator(method="statevector")
    qc = get_qaoa_circuit(N, terms, betas, gammas)
    return np.asarray(backend.run(qc).result().get_statevector())


def _run_qiskit_mixer(N, betas):
    backend = AerSimulator(method="statevector")
    qc = QuantumCircuit(N)
    for beta in betas:
        qc.rx(2 * beta, list(range(N)))
    qc.save_state()
    return np.asarray(backend.run(qc).result().get_statevector())


def _check_results(res, expected):
    for name, sv in res.items():
        assert np.allclose(sv, expected), f"results from simulator {name} do not match with qiskit"


@pytest.mark.parametrize("N, p", [(5, 1), (5, 2)])
def test_mixer(N, p):
    """
    apply uniform Rx on |0> state
    """
    gammas = [0] * p
    betas = np.random.uniform(0, np.pi, p)
    costs = np.zeros(2**N)
    sv0 = np.array([1] + [0] * (2**N - 1))

    res = _run_all_simulators(N, costs, gammas, betas, sv0=sv0)
    sv_qiskit = _run_qiskit_mixer(N, betas)

    _check_results(res, sv_qiskit)


@pytest.mark.parametrize("N, p", [(3, 1), (3, 2)])
def test_phase_operator(N, p):
    """
    apply phase operator on uniform superposition
    """
    gammas = np.random.uniform(0, np.pi, p)
    betas = [0] * p
    terms = [[0, 1, 2], [0]]
    offset = 2
    f = partial(energy_vals_from_bitstring_general, terms=terms, offset=offset)
    precomputed_energies = precompute_energies(f, N) - offset

    res = _run_all_simulators(N, precomputed_energies, gammas, betas)
    sv_qiskit = _run_qiskit(N, terms, gammas, betas)

    _check_results(res, sv_qiskit)


@pytest.mark.parametrize("N, p", [(3, 1), (3, 2)])
def test_phase_operator_labs(N, p):
    """
    apply phase operator on uniform superposition (using cost from LABS)
    """
    gammas = np.random.uniform(0, np.pi, p)
    betas = [0] * p
    terms, offset = get_energy_term_indices(N)
    precomputed_energies = -(N**2) / (2 * get_precomputed_labs_merit_factors(N)) - offset

    res = _run_all_simulators(N, precomputed_energies, gammas, betas)
    sv_qiskit = _run_qiskit(N, terms, gammas, betas)

    _check_results(res, sv_qiskit)


def test_against_qiskit_p_1():
    beta = np.random.uniform(0, np.pi)
    gamma = np.random.uniform(0, np.pi)

    N = 5
    terms, offset = get_energy_term_indices(N)
    precomputed_energies = -(N**2) / (2 * get_precomputed_labs_merit_factors(N)) - offset

    res = _run_all_simulators(N, precomputed_energies, [gamma], [beta])
    sv_qiskit = _run_qiskit(N, terms, [gamma], [beta])

    _check_results(res, sv_qiskit)


def test_against_qiskit_p_3():
    p = 3
    N = 5
    terms, offset = get_energy_term_indices(N)
    precomputed_energies = -(N**2) / (2 * get_precomputed_labs_merit_factors(N)) - offset

    for _ in range(5):
        betas = np.random.uniform(0, np.pi, p)
        gammas = np.random.uniform(0, np.pi, p)

        res = _run_all_simulators(N, precomputed_energies, gammas, betas)
        sv_qiskit = _run_qiskit(N, terms, gammas, betas)

        _check_results(res, sv_qiskit)
