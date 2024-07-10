import numpy as np

"""
This file implements a dumber version of the QAOA proxy algorithm for MaxCut from:
https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.6.023171

But using sillier approxmations of distributions
"""


# P(c') from paper but dumber
def prob_cost_dumb(cost: int, num_constraints: int) -> float:
    return 4 / ((num_constraints + 1) ** 2) * min(cost + 1, num_constraints + 1 - cost)


# N(c') from paper but dumber
def number_with_cost_proxy_dumb(cost: int, num_constraints: int, num_qubits: int) -> float:
    scale = 1 << num_qubits
    return prob_cost_dumb(cost, num_constraints) * scale


# N(c'; d, c) from paper but dumber (IN FACT, SO DUMB THAT IT'S SUPER WRONG! SHOULD USE A CONE OR SOMETHING INSTEAD)
def number_of_costs_at_distance_proxy_dumb(cost_1: int, cost_2: int, distance: int, num_constraints: int, num_qubits: int, prob_edge: float = 0.5) -> float:
    cost_diff = abs(cost_1 - cost_2)
    num_w_cost_1 = number_with_cost_proxy_dumb(cost_1, num_constraints, num_qubits)
    simplex_value = num_w_cost_1 * (1 - cost_diff / max(num_constraints - cost_diff, cost_diff) - distance / num_qubits)
    return max(simplex_value, 0)


# Computes the sum inside the for loop of Algorithm 1 in paper using dumb approximations
def compute_amplitude_sum_dumb(prev_amplitudes: np.ndarray, gamma: float, beta: float, cost_1: int, num_constraints: int, num_qubits: int) -> complex:
    sum = 0
    for cost_2 in range(num_constraints + 1):
        for distance in range(num_qubits + 1):
            # Should I np-ify all of the stuff here?
            beta_factor = (np.cos(beta) ** (num_qubits - distance)) * ((-1j * np.sin(beta)) ** distance)
            gamma_factor = np.exp(-1j * gamma * cost_2)
            num_costs_at_distance = number_of_costs_at_distance_proxy_dumb(cost_1, cost_2, distance, num_constraints, num_qubits)
            sum += beta_factor * gamma_factor * prev_amplitudes[cost_2] * num_costs_at_distance
    return sum


# TODO: What if instead of optimizing expectation proxy we instead optimize high cost amplitudes (using e.g. exponential weighting)
# Algorithm 1 from paper using dumb approximations
# num_constraints = number of edges, and num_qubits = number of vertices
def QAOA_proxy_dumb(p: int, gamma: np.ndarray, beta: np.ndarray, num_constraints: int, num_qubits: int):
    num_costs = num_constraints + 1
    amplitude_proxies = np.zeros([p + 1, num_costs], dtype=complex)
    init_amplitude = np.sqrt(1 / (1 << num_qubits))
    for i in range(num_costs):
        amplitude_proxies[0][i] = init_amplitude

    for current_depth in range(1, p + 1):
        for cost_1 in range(num_costs):
            amplitude_proxies[current_depth][cost_1] = compute_amplitude_sum_dumb(
                amplitude_proxies[current_depth - 1], gamma[current_depth - 1], beta[current_depth - 1], cost_1, num_constraints, num_qubits
            )

    expected_proxy = 0
    for cost in range(num_costs):
        expected_proxy += number_with_cost_proxy_dumb(cost, num_constraints, num_qubits) * (abs(amplitude_proxies[p][cost]) ** 2) * cost

    return amplitude_proxies, expected_proxy
