###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import numpy as np
import pytest
from functools import partial
from qiskit_aer import AerSimulator
from itertools import combinations
from qokit.sk import sk_obj, get_sk_terms, get_random_J
from qokit.maxcut import maxcut_obj
from qokit.qaoa_objective_sk import get_qaoa_sk_objective
from qokit.utils import brute_force, precompute_energies
from qokit.parameter_utils import get_sk_gamma_beta
from qokit.fur import get_available_simulators, get_available_simulator_names


rng = np.random.default_rng(seed=42)

SIMULATORS = get_available_simulators("x")
simulators_to_run_names = get_available_simulator_names("x") + ["qiskit"]


def test_sk_obj(N=5):
    J = get_random_J(N=N)

    def sk_obj_simple(x, J):
        obj = 0
        for i in range(N):
            for j in range(i, N):
                obj += J[i, j] * (2 * x[i] - 1) * (2 * x[j] - 1)
        return 2 * obj / np.sqrt(N)

    x = rng.choice([0, 1], N)
    assert np.isclose(sk_obj(x, J), sk_obj_simple(x, J))


def test_parameter_normalization(N=20):
    J = get_random_J(N=N)

    f = get_qaoa_sk_objective(N, 1, J=J, parameterization="gamma beta", simulator=simulators_to_run_names[0], objective="expectation")
    assert np.isclose(f([1], [np.pi / 8]) / N, 0.3, atol=1e-1)


def test_variance_normalization(N=1000):
    J = get_random_J(N=N)

    terms = get_sk_terms(J)
    terms_0 = [x[0] for x in terms]
    var = np.var(terms_0)
    print("var: ", var)

    assert np.isclose(N * var, 1, atol=1e-2)


@pytest.mark.parametrize("simulator", simulators_to_run_names)
def test_energy_pre_optimized(simulator, N=8):
    J = np.array(
        [
            [0.00000000e00, 1.15336475e00, -5.85084225e-01, 7.12352577e-01, 3.68067836e-01, -1.13088622e00, -8.64623738e-02, -3.77646427e-01],
            [1.15336475e00, 0.00000000e00, -9.13387355e-02, -7.66210270e-01, -7.19919730e-01, -8.00833339e-01, 2.33528912e-01, -6.65389692e-01],
            [-5.85084225e-01, -9.13387355e-02, 0.00000000e00, 2.68564121e-01, 9.91582576e-01, -9.31645738e-01, 2.33491255e-02, -7.41120517e-01],
            [7.12352577e-01, -7.66210270e-01, 2.68564121e-01, 0.00000000e00, 9.70605475e-01, 1.02954907e00, -3.94682792e-01, -3.25847934e-04],
            [3.68067836e-01, -7.19919730e-01, 9.91582576e-01, 9.70605475e-01, 0.00000000e00, -4.07804975e-01, 2.67471661e-01, -1.60992252e-02],
            [-1.13088622e00, -8.00833339e-01, -9.31645738e-01, 1.02954907e00, -4.07804975e-01, 0.00000000e00, 8.64544538e-01, 7.83512686e-01],
            [-8.64623738e-02, 2.33528912e-01, 2.33491255e-02, -3.94682792e-01, 2.67471661e-01, 8.64544538e-01, 0.00000000e00, -1.23662276e-01],
            [-3.77646427e-01, -6.65389692e-01, -7.41120517e-01, -3.25847934e-04, -1.60992252e-02, 7.83512686e-01, -1.23662276e-01, 0.00000000e00],
        ]
    )

    p = 8
    gamma = np.array([0.2268, 0.4162, 0.4332, 0.4608, 0.4818, 0.5179, 0.5717, 0.6393])
    beta = np.array([0.6151, 0.4906, 0.4244, 0.3780, 0.3224, 0.2606, 0.1884, 0.1030])

    # Precomputed expected energy with the above fixed parameters: [4.207818619693583]
    expected_energy = [4.207818619693583]

    qaoa_objectives = get_qaoa_sk_objective(N, p, J=J, parameterization="gamma beta", simulator=simulator, objective="expectation")(gamma, beta)
    assert np.allclose(qaoa_objectives, expected_energy)


@pytest.mark.parametrize("simulator", simulators_to_run_names)
def test_sk_qaoa_convergence_with_p(simulator, N=12):
    J = get_random_J(N=N)

    obj = partial(sk_obj, J=J)
    precomputed_energies = precompute_energies(obj, N)
    max_energy = np.max(precomputed_energies)

    last_objective = 0.0
    last_overlap = 0.0

    for p in range(1, 18):
        gamma, beta = get_sk_gamma_beta(p)
        qaoa_objective = get_qaoa_sk_objective(
            N, p, J=J, precomputed_energies=precomputed_energies, simulator=simulator, parameterization="gamma beta", objective="expectation"
        )(2 * gamma / np.sqrt(N), beta)
        qaoa_overlap = get_qaoa_sk_objective(
            N, p, J=J, precomputed_energies=precomputed_energies, simulator=simulator, parameterization="gamma beta", objective="overlap"
        )(2 * gamma / np.sqrt(N), beta)
        current_objective = qaoa_objective / max_energy
        current_overlap = 1 - qaoa_overlap
        assert current_objective < last_objective
        assert current_overlap > last_overlap
        last_objective = current_objective
        last_overlap = current_overlap


def test_sk_qaoa_obj_consistency_across_simulators(N=8):
    J = get_random_J(N=N)

    for p in range(1, 18):
        gamma, beta = get_sk_gamma_beta(p)
        for objective in ["expectation", "overlap"]:
            qaoa_objectives = [
                get_qaoa_sk_objective(N, p, J=J, parameterization="gamma beta", simulator=simulator, objective=objective)(gamma, beta)
                for simulator in simulators_to_run_names
            ]
            assert np.all(np.isclose(qaoa_objectives, qaoa_objectives[0]))


@pytest.mark.parametrize("simulator", simulators_to_run_names)
def test_sk_qaoa_obj_fixed_angles_and_precomputed_energies(simulator, N=10):
    max_p = 11
    J = get_random_J(N=N)
    obj = partial(sk_obj, J=J)
    precomputed_energies = precompute_energies(obj, N)
    for p in range(1, max_p + 1):
        gamma, beta = get_sk_gamma_beta(p)
        f1 = get_qaoa_sk_objective(N, p, J=J, parameterization="gamma beta", simulator=simulator)
        f2 = get_qaoa_sk_objective(N, p, J=J, precomputed_energies=precomputed_energies, parameterization="gamma beta", simulator=simulator)
        e1 = f1(gamma, beta)
        e2 = f2(gamma, beta)
        assert np.isclose(e1, e2)


@pytest.mark.parametrize("simclass", SIMULATORS)
def test_sk_precompute(simclass, N=4):
    J = get_random_J(N=N)

    precomputed_energies = precompute_energies(sk_obj, N, J)
    terms = get_sk_terms(J)
    sim = simclass(N, terms=terms)
    cuts = sim.get_cost_diagonal()
    assert np.allclose(precomputed_energies, cuts, atol=1e-6)


def test_sk_maxcut_bruteforce(N=10):
    J = get_random_J(N=N)

    obj_maxcut = partial(maxcut_obj, w=J)
    optimal_maxcut, x_maxcut = brute_force(obj_maxcut, num_variables=N, function_takes="bits")
    obj_sk = partial(sk_obj, J=J)
    optimal_sk, x_sk = brute_force(obj_sk, num_variables=N, minimize=True, function_takes="bits")
    x_maxcut_comp = 1 - x_maxcut

    assert np.allclose(x_maxcut, x_sk) or np.allclose(x_maxcut_comp, x_sk)


@pytest.mark.parametrize("simulator", simulators_to_run_names)
def test_overlap_sk(simulator, N=4):
    J = get_random_J(N=N)
    p = 1
    beta = [rng.uniform(0, 1)]
    gamma = [rng.uniform(0, 1)]

    obj = partial(sk_obj, J=J)
    precomputed_energies = precompute_energies(obj, N)

    f1 = get_qaoa_sk_objective(N, p, J=J, precomputed_energies=precomputed_energies, simulator=simulator, parameterization="gamma beta", objective="overlap")
    f2 = get_qaoa_sk_objective(N, p, J=J, parameterization="gamma beta", simulator=simulator, objective="overlap")

    assert np.isclose(f1(gamma, beta), f2(gamma, beta))
    assert np.isclose(f1([0], [0]), f2([0], [0]))

    maxval = precomputed_energies.max()
    bitstring_loc = (precomputed_energies == maxval).nonzero()
    assert len(bitstring_loc) == 1
    bitstring_loc = bitstring_loc[0]
    assert np.isclose(1 - f1([0], [0]), len(bitstring_loc) / len(precomputed_energies))
