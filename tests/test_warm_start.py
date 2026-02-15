###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import numpy as np
import networkx as nx
import qokit
from qokit.parameter_utils import get_fixed_gamma_beta
from qokit.warm_start import WSSolver, maxcut_qubo_from_G, WSSolverQUBO, get_terms_from_QUBO
from qokit.maxcut import get_maxcut_terms
from qokit.qaoa_objective_maxcut import get_qaoa_maxcut_objective
import pytest
import sys


def test_p0_objective():
    n_v = 10
    graph_degree = 3
    graph_seed = 0
    G = nx.random_regular_graph(graph_degree, n_v, seed=graph_seed)
    solver = WSSolver(
        graph=G,
        graph_degree=graph_degree,
        graph_seed=graph_seed,
    )
    theta = np.random.rand(n_v)
    f_value = solver.get_p0_cut(theta)
    expH, _, _, _ = solver.get_p0_std_quantities(theta)
    assert np.isclose(expH, f_value)


def test_ws_degeneracy():
    n_v = 10
    graph_degree = 3
    graph_seed = 0
    G = nx.random_regular_graph(graph_degree, n_v, seed=graph_seed)
    solver = WSSolver(
        graph=G,
        graph_degree=graph_degree,
        graph_seed=graph_seed,
    )
    theta = np.pi / 2 * np.ones(n_v)
    p = 3
    ws_qaoa_energy = solver.run_ws_qaoa(p=p, theta=theta, check_saved_result=False)
    qaoa_energy = solver.run_standard_qaoa(p=p)
    assert np.isclose(ws_qaoa_energy, qaoa_energy)


def test_batch_obj_grad():
    N = 10
    G = nx.random_regular_graph(3, N, seed=0)
    ws_solver = WSSolver(G)

    theta1 = np.random.rand(N)
    theta2 = np.random.rand(N)

    assert np.allclose(np.asarray([ws_solver.bm_objective(theta1), ws_solver.bm_objective(theta2)]), ws_solver.bm_objective_batch(np.vstack([theta1, theta2])))
    assert np.allclose(np.asarray([ws_solver.bm_gradient(theta1), ws_solver.bm_gradient(theta2)]), ws_solver.bm_gradient_batch(np.vstack([theta1, theta2])))
    assert np.allclose(
        np.asarray([ws_solver.p0_theta_objective(theta1), ws_solver.p0_theta_objective(theta2)]),
        ws_solver.p0_theta_objective_batch(np.vstack([theta1, theta2])),
    )
    assert np.allclose(
        np.asarray([ws_solver.p0_theta_grad(theta1), ws_solver.p0_theta_grad(theta2)]), ws_solver.p0_theta_grad_batch(np.vstack([theta1, theta2]))
    )

    local_bit = np.random.randint(0, 2, size=N)
    assert np.allclose(
        np.asarray([ws_solver.abid_gradient(theta1, local_bit), ws_solver.abid_gradient(theta2, local_bit)]),
        ws_solver.abid_gradient_batch(np.vstack([theta1, theta2]), local_bit),
    )
    assert np.allclose(
        np.asarray([ws_solver.abid_objective(theta1, local_bit), ws_solver.abid_objective(theta2, local_bit)]),
        ws_solver.abid_objective_batch(np.vstack([theta1, theta2]), local_bit),
    )


############################
@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="Fast c/c++ simulator should be installed")
def test_maxcut_qaoa():
    N = 10
    G = nx.random_regular_graph(3, N, seed=1)

    Q = maxcut_qubo_from_G(G)
    qubo_ws_solver = WSSolverQUBO(Q)
    terms = get_terms_from_QUBO(Q)

    p = 2
    gamma, beta = get_fixed_gamma_beta(3, p)
    simclass = qokit.fur.choose_simulator_xz(name="c")
    sim = simclass(N, terms=terms)
    cost = sim.get_cost_diagonal()
    best_cut = np.max(cost)

    terms_maxcut = get_maxcut_terms(G)
    sim_maxcut = simclass(N, terms=terms_maxcut)
    cost_maxcut = sim_maxcut.get_cost_diagonal()
    mean_cut_maxcut = np.mean(cost_maxcut)
    maxcut_obj = get_qaoa_maxcut_objective(N, p, G=G, parameterization="gamma beta", simulator="c", objective="expectation")(gamma, beta)

    _result = sim.simulate_ws_qaoa(list(np.asarray(gamma)), list(np.asarray(beta)), np.ones(N) * np.pi / 2)
    qokit_energy = sim.get_expectation(_result)

    qubo_energy = qubo_ws_solver.run_standard_qaoa(gamma, beta)

    ws_solver = WSSolver(G)
    ws_energy = ws_solver.run_standard_qaoa(p)
    assert np.isclose(qokit_energy, qubo_energy)
    assert np.isclose(ws_energy, qubo_energy)
    assert mean_cut_maxcut < ws_energy
    assert maxcut_obj < ws_energy
