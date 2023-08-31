from qokit.portfolio_optimization import get_problem, get_problem_H, get_problem_H_bf, po_obj_func, portfolio_brute_force, get_sk_ini
from qokit.utils import precompute_energies, reverse_array_index_bit_order
from qokit.qaoa_circuit_portfolio import (
    circuit_measurement_function,
    generate_dicke_state_fast,
    get_energy_expectation_sv,
    get_qaoa_circuit,
    get_parameterized_qaoa_circuit,
)
from qokit.fur import get_available_simulator_names, choose_simulator_xyring
from qokit.qaoa_objective_portfolio import get_qaoa_portfolio_objective
import qokit
import qokit
import numpy as np
import pytest
from qiskit import execute, Aer
from functools import reduce

simulators_to_run = get_available_simulator_names("xyring")


def test_portfolio_precompute():
    N = 6
    K = 5
    q = 0.5
    seed = None
    scale = 1
    po_problem = get_problem(N=N, K=K, q=q, seed=seed, pre=scale)

    po_obj = po_obj_func(po_problem)
    precomputed_energies = reverse_array_index_bit_order(precompute_energies(po_obj, N))

    H_diag = np.diag(get_problem_H(po_problem))
    np.allclose(precomputed_energies, H_diag)


##################################################


@pytest.mark.parametrize("simname_ring", simulators_to_run)
def test_portfolio_furandqiskit(simname_ring):
    K = 3
    q = 0.5
    seed = 2053
    scale = 10
    for N in [4, 5]:
        po_problem = get_problem(N=N, K=K, q=q, seed=seed, pre=scale)
        po_obj = po_obj_func(po_problem)
        precomputed_energies = reverse_array_index_bit_order(precompute_energies(po_obj, N))
        simclass = choose_simulator_xyring(name=simname_ring)
        sim = simclass(N, po_problem["scale"] * precomputed_energies)
        for p in [1, 2]:
            for T in [1, 2]:
                gammas = np.random.rand(p)
                betas = np.random.rand(p)
                obj = circuit_measurement_function(
                    po_problem=po_problem,
                    p=p,
                    ini="dicke",
                    mixer="trotter_ring",
                    T=T,
                    ini_state=None,
                    save_state=True,
                    n_trials=1024,  # number of shots if save_state is False
                    minus=False,
                )
                ini_x = np.concatenate((gammas, betas), axis=0)
                f1 = obj(ini_x)

                sv = sim.simulate_qaoa(2 * gammas, 2 * betas, sv0=generate_dicke_state_fast(N, K), n_trotters=T)
                probs = sim.get_probabilities(sv)
                f2 = probs.dot(precomputed_energies)

                assert np.isclose(f1, f2)


@pytest.mark.parametrize("simname", simulators_to_run)
def test_portfolio_qokitandqiskit(simname):
    p = 1
    K = 3
    q = 0.5
    seed = 1
    backend = Aer.get_backend("statevector_simulator")
    for N in [4, 5]:
        for T in [1, 2]:
            po_problem = get_problem(N=N, K=K, q=q, seed=seed, pre="rule")
            x0 = np.random.rand(2 * p)
            qaoa_obj_qokit = get_qaoa_portfolio_objective(po_problem=po_problem, p=p, simulator=simname, ini="dicke", mixer="trotter_ring", T=1)
            qaoa_obj_qiskit = get_qaoa_portfolio_objective(po_problem=po_problem, p=p, simulator="qiskit", ini="dicke", mixer="trotter_ring", T=1)
            assert np.allclose(qaoa_obj_qiskit(x0), qaoa_obj_qokit(x0))

            qc = get_qaoa_circuit(po_problem, gammas=x0[:p] / 2, betas=x0[p:] / 2, depth=p)
            result = execute(qc, backend).result()
            sv1 = reverse_array_index_bit_order(result.get_statevector())
            #####
            parameterized_qc = get_parameterized_qaoa_circuit(po_problem, depth=p)
            qc2 = parameterized_qc.bind_parameters(np.hstack([x0[p:] / 2, x0[:p] / 2]))
            result = execute(qc2, backend).result()
            sv2 = reverse_array_index_bit_order(result.get_statevector())
            assert np.allclose(sv1, sv2)
            assert np.allclose(get_energy_expectation_sv(po_problem, sv1), qaoa_obj_qiskit(x0))


@pytest.mark.parametrize("simname", simulators_to_run)
def test_qaoa_portfolio_objective_qiskit_simulator(simname):
    K = 3
    q = 0.5
    seed = 2053
    scale = 10
    for N in [4, 5]:
        po_problem = get_problem(N=N, K=K, q=q, seed=seed, pre=scale)
        for p in [1, 2]:
            for T in [1, 2]:
                qaoa_obj = get_qaoa_portfolio_objective(po_problem=po_problem, p=p, ini="dicke", mixer="trotter_ring", T=T, simulator=simname)
                qaoa_obj2 = get_qaoa_portfolio_objective(po_problem=po_problem, p=p, simulator="qiskit", ini="dicke", mixer="trotter_ring", T=T)
                x0 = np.random.rand(p * 2)
                assert np.isclose(qaoa_obj(x0), qaoa_obj2(x0))


def test_get_problem_H():
    K = 3
    q = 0.5
    seed = 2053
    scale = 10
    for N in [4, 5]:
        po_problem = get_problem(N=N, K=K, q=q, seed=seed, pre=scale)

        H_problem = get_problem_H(po_problem)
        H_problem_bf = get_problem_H_bf(po_problem)
        assert np.allclose(H_problem, H_problem_bf)


def test_portfolio_AR():
    po_problem = get_problem(N=6, K=3, q=0.5, seed=1, pre="rule")
    p = 1
    qaoa_obj = get_qaoa_portfolio_objective(po_problem=po_problem, p=p, ini="dicke", mixer="trotter_ring", T=1, simulator="python")
    best_portfolio = portfolio_brute_force(po_problem, return_bitstring=False)
    x0 = get_sk_ini(p=p)
    po_energy = qaoa_obj(x0).real
    po_ar = (po_energy - best_portfolio[1]) / (best_portfolio[0] - best_portfolio[1])
    # a problem with known AR > 0.7564
    assert po_ar > 0.75


def test_best_bitstring():
    N = 6
    po_problem = get_problem(N=N, K=3, q=0.5, seed=1, pre="rule")
    bf_result = portfolio_brute_force(po_problem, return_bitstring=True)
    precomputed_optimal_bitstrings = bf_result[1].reshape(1, -1)
    assert precomputed_optimal_bitstrings.shape[1] == N
    bitstring_loc = np.array([reduce(lambda a, b: 2 * a + b, x) for x in precomputed_optimal_bitstrings])
    assert len(bitstring_loc) == 1
    po_obj = po_obj_func(po_problem)
    precomputed_energies = reverse_array_index_bit_order(precompute_energies(po_obj, N)).real
    assert np.isclose(precomputed_energies[bitstring_loc[0]], bf_result[0])
