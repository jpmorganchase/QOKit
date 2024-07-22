import numpy as np
import math
import typing
import time
import scipy
from scipy.stats import binom, multinomial

"""
This file implements the QAOA proxy algorithm for MaxCut from:
https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.6.023171
"""


# P(c') from paper
def prob_cost_paper(cost: int, num_constraints: int, prob_edge: float = 0.5) -> float:
    return binom.pmf(cost, num_constraints, prob_edge)


# N(c') from paper
def number_with_cost_paper_proxy(cost: int, num_constraints: int, num_qubits: int, prob_edge: float = 0.5) -> float:
    scale = 1 << num_qubits
    return prob_cost_paper(cost, num_constraints, prob_edge) * scale


# P(b, c'-b, c-b | d) from paper
def prob_common_at_distance_paper(num_constraints: int, common_constraints: int, cost_1: int, cost_2: int, distance: int) -> float:
    prob_same = (math.comb(num_constraints - distance, 2) + math.comb(distance, 2)) / math.comb(num_constraints, 2)
    prob_neither = prob_same / 2
    prob_both = prob_neither
    prob_one = (1 - prob_neither - prob_both) / 2
    return multinomial.pmf(
        [common_constraints, cost_1 - common_constraints, cost_2 - common_constraints, num_constraints + common_constraints - (cost_1 + cost_2)],
        num_constraints,
        [prob_both, prob_one, prob_one, prob_neither],
    )


# N(c'; d, c) from paper
def number_of_costs_at_distance_paper_proxy(cost_1: int, cost_2: int, distance: int, num_constraints: int, num_qubits: int, prob_edge: float = 0.5) -> float:
    sum = 0
    for common_constraints in range(max(0, cost_1 + cost_2 - num_constraints), min(cost_1, cost_2) + 1):
        sum += prob_common_at_distance_paper(num_constraints, common_constraints, cost_1, cost_2, distance)

    p_cost = prob_cost_paper(cost_1, num_constraints, prob_edge)
    return (math.comb(num_qubits, distance) / p_cost) * sum


# Computes the sum inside the for loop of Algorithm 1 in paper
def compute_amplitude_sum_paper(prev_amplitudes: np.ndarray, gamma: float, beta: float, cost_1: int, num_constraints: int, num_qubits: int) -> complex:
    sum = 0
    for cost_2 in range(num_constraints + 1):
        for distance in range(num_qubits + 1):
            # Should I np-ify all of the stuff here?
            beta_factor = (np.cos(beta) ** (num_qubits - distance)) * ((-1j * np.sin(beta)) ** distance)
            gamma_factor = np.exp(-1j * gamma * cost_2)
            num_costs_at_distance = number_of_costs_at_distance_paper_proxy(cost_1, cost_2, distance, num_constraints, num_qubits)
            sum += beta_factor * gamma_factor * prev_amplitudes[cost_2] * num_costs_at_distance
    return sum


# TODO: What if instead of optimizing expectation proxy we instead optimize high cost amplitudes (using e.g. exponential weighting)
# Algorithm 1 from paper
# num_constraints = number of edges, and num_qubits = number of vertices
def QAOA_paper_proxy(p: int, gamma: np.ndarray, beta: np.ndarray, num_constraints: int, num_qubits: int, terms_to_drop_in_expectation: int = 0):
    num_costs = num_constraints + 1
    amplitude_proxies = np.zeros([p + 1, num_costs], dtype=complex)
    init_amplitude = np.sqrt(1 / (1 << num_qubits))
    for i in range(num_costs):
        amplitude_proxies[0][i] = init_amplitude

    for current_depth in range(1, p + 1):
        for cost_1 in range(num_costs):
            amplitude_proxies[current_depth][cost_1] = compute_amplitude_sum_paper(
                amplitude_proxies[current_depth - 1], gamma[current_depth - 1], beta[current_depth - 1], cost_1, num_constraints, num_qubits
            )

    expected_proxy = 0
    for cost in range(terms_to_drop_in_expectation, num_costs):
        expected_proxy += number_with_cost_paper_proxy(cost, num_constraints, num_qubits) * (abs(amplitude_proxies[p][cost]) ** 2) * cost

    return amplitude_proxies, expected_proxy


def inverse_paper_proxy_objective_function(num_constraints: int, num_qubits: int, p: int, expectations: list[np.ndarray] | None) -> typing.Callable:
    def inverse_objective(*args) -> float:
        gamma, beta = args[0][:p], args[0][p:]
        _, expectation = QAOA_paper_proxy(p, gamma, beta, num_constraints, num_qubits)
        current_time = time.time()

        if expectations is not None:
            expectations.append((current_time, expectation))

        return -expectation

    return inverse_objective


def QAOA_paper_proxy_run(
    num_constraints: int,
    num_qubits: int,
    p: int,
    init_gamma: np.ndarray,
    init_beta: np.ndarray,
    optimizer_method: str = "COBYLA",
    optimizer_options: dict | None = None,
    expectations: list[np.ndarray] | None = None,
) -> dict:
    init_freq = np.hstack([init_gamma, init_beta])

    start_time = time.time()
    result = scipy.optimize.minimize(
        inverse_paper_proxy_objective_function(num_constraints, num_qubits, p, expectations),
        init_freq,
        args=(),
        method=optimizer_method,
        options=optimizer_options,
    )
    # the above returns a scipy optimization result object that has multiple attributes
    # result.x gives the optimal solutionsol.success #bool whether algorithm succeeded
    # result.message #message of why algorithms terminated
    # result.nfev is number of iterations used (here, number of QAOA calls)
    end_time = time.time()

    def make_time_relative(input: tuple[float, float]) -> tuple[float, float]:
        time, x = input
        return (time - start_time, x)

    if expectations is not None:
        expectations = list(map(make_time_relative, expectations))

    gamma, beta = result.x[:p], result.x[p:]
    _, expectation = QAOA_paper_proxy(p, gamma, beta, num_constraints, num_qubits)

    return {
        "gamma": gamma,
        "beta": beta,
        "expectation": expectation,
        "runtime": end_time - start_time,  # measured in seconds
        "num_QAOA_calls": result.nfev,  # Calls to the proxy of course
        "classical_opt_success": result.success,
        "scipy_opt_message": result.message,
    }
