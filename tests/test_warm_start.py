###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import numpy as np
import networkx as nx
import qokit
from qokit.fur import get_available_simulator_names
from qokit.parameter_utils import get_fixed_gamma_beta
from qokit.warm_start import WSSolver, maxcut_qubo_from_G, WSSolverQUBO, get_terms_from_QUBO
from qokit.maxcut import get_maxcut_terms
from qokit.qaoa_objective_maxcut import get_qaoa_maxcut_objective
import pytest
import sys

simulators_to_run = get_available_simulator_names("xz") 

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
    expH = solver.rws_objective(theta)
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
    gamma, beta = get_fixed_gamma_beta(graph_degree, p)
    ws_qaoa_energy = solver.run_ws_qaoa(p=p, gamma=gamma, beta=beta, theta=theta)
    qaoa_energy = solver.run_standard_qaoa(p=p)
    assert np.isclose(ws_qaoa_energy, qaoa_energy)


def test_batch_obj_grad():
    N = 10
    G = nx.random_regular_graph(3, N, seed=0)
    ws_solver = WSSolver(G)

    theta1 = np.random.rand(N)
    theta2 = np.random.rand(N)

    assert np.allclose(np.asarray([ws_solver.bm_objective(theta1), ws_solver.bm_objective(theta2)]), ws_solver.bm_objective(np.vstack([theta1, theta2])))
    assert np.allclose(np.asarray([ws_solver.bm_gradient(theta1), ws_solver.bm_gradient(theta2)]), ws_solver.bm_gradient(np.vstack([theta1, theta2])))
    assert np.allclose(
        np.asarray([ws_solver.rws_objective(theta1), ws_solver.rws_objective(theta2)]),
        ws_solver.rws_objective(np.vstack([theta1, theta2])),
    )
    assert np.allclose(
        np.asarray([ws_solver.rws_grad(theta1), ws_solver.rws_grad(theta2)]), ws_solver.rws_grad(np.vstack([theta1, theta2]))
    )

@pytest.mark.parametrize("simulator", simulators_to_run)
def test_ws_qaoa_better_than_qaoa(simulator):
    N = 10
    G = nx.random_regular_graph(3, N, seed=1)
    terms_maxcut = get_maxcut_terms(G)
    ws_solver = WSSolver(G)

    theta, p0_energy = ws_solver.optimize_theta(
                    objective = 'rws', 
                    optimizer = 'ADAM', 
                    global_alpha = False, 
                    trials = 100, 
                    lamd = 0.6, 
                    )

    p = 2
    ws_gamma, ws_beta = ws_solver.get_ws_qaoa_para(p)
    simclass = qokit.fur.choose_simulator_xz(name=simulator)
    sim = simclass(N, terms=terms_maxcut)
    _result = sim.simulate_ws_qaoa(list(np.asarray(ws_gamma)), list(np.asarray(ws_beta)), theta)
    ws_energy = sim.get_expectation(_result)
    
    sim_maxcut = simclass(N, terms=terms_maxcut)
    cost_maxcut = sim_maxcut.get_cost_diagonal()
    mean_cut_maxcut = np.mean(cost_maxcut)

    gamma, beta = get_fixed_gamma_beta(3,p)
    qaoa_energy = get_qaoa_maxcut_objective(N, p, G=G, parameterization="gamma beta", simulator=simulator, objective="expectation")(gamma, beta)
    
    assert ws_energy > p0_energy
    assert mean_cut_maxcut < ws_energy
    assert qaoa_energy < ws_energy


def test_ws_qaoa_p2_better_than_p1():
    """Test that WS-QAOA at p=2 yields higher energy than p=1 (notebook example)."""
    n_v = 16
    graph_degree = 3
    graph_seed = 0
    G = nx.random_regular_graph(graph_degree, n_v, seed=graph_seed)

    solver = WSSolver(
        graph=G,
        graph_degree=graph_degree,
        graph_seed=graph_seed,
    )

    lamd = 0.6
    theta, p0_energy = solver.optimize_theta(
        objective='rws',
        optimizer='ADAM',
        trials=100,
        lamd=lamd,
    )

    gamma1, beta1 = solver.get_ws_qaoa_para(p=1)
    energy_p1 = solver.run_ws_qaoa(gamma=gamma1, beta=beta1, theta=theta)

    gamma2, beta2 = solver.get_ws_qaoa_para(p=2)
    energy_p2 = solver.run_ws_qaoa(gamma=gamma2, beta=beta2, theta=theta)

    assert energy_p2 > energy_p1, f"p=2 energy ({energy_p2}) should exceed p=1 energy ({energy_p1})"
    assert energy_p1 > p0_energy, f"p=1 energy ({energy_p1}) should exceed p0 energy ({p0_energy})"
