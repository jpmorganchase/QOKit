import numpy as np
import math
from scipy.stats import binom, multinomial

"""
This file implements the QAOA proxy algorithm for MaxCut from:
https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.6.023171
"""


# P(c') from paper
def prob_cost(cost: int, num_constraints: int, prob_edge: float = 0.5) -> float:
    return binom.pmf(cost, num_constraints, prob_edge)


# N(c') from paper
def number_with_cost_proxy(cost: int, num_constraints: int, num_qubits: int, prob_edge: float = 0.5) -> float:
    scale = 1 << num_qubits
    return prob_cost(cost, num_constraints, prob_edge) * scale


# P(b, c'-b, c-b | d) from paper
def prob_common_at_distance(num_constraints: int, common_constraints: int, cost_1: int, cost_2: int, distance: int) -> float:
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
def number_of_costs_at_distance_proxy(cost_1: int, cost_2: int, distance: int, num_constraints: int, num_qubits: int, prob_edge: float = 0.5) -> float:
    sum = 0
    for common_constraints in range(max(0, cost_1 + cost_2 - num_constraints), min(cost_1, cost_2) + 1):
        sum += prob_common_at_distance(num_constraints, common_constraints, cost_1, cost_2, distance)

    p_cost = prob_cost(cost_1, num_constraints, prob_edge)
    return (math.comb(num_qubits, distance) / p_cost) * sum


# Computes the sum inside the for loop of Algorithm 1 in paper
def compute_amplitude_sum(prev_amplitudes: np.ndarray, gamma: float, beta: float, cost_1: int, num_constraints: int, num_qubits: int) -> complex:
    sum = 0
    for cost_2 in range(num_constraints + 1):
        for distance in range(num_qubits + 1):
            # Should I np-ify all of the stuff here?
            beta_factor = (np.cos(beta) ** (num_qubits - distance)) * ((-1j * np.sin(beta)) ** distance)
            gamma_factor = np.exp(-1j * gamma * cost_2)
            num_costs_at_distance = number_of_costs_at_distance_proxy(cost_1, cost_2, distance, num_constraints, num_qubits)
            sum += beta_factor * gamma_factor * prev_amplitudes[cost_2] * num_costs_at_distance
    return sum


# TODO: What if instead of optimizing expectation proxy we instead optimize high cost amplitudes (using e.g. exponential weighting)
# Algorithm 1 from paper
# num_constraints = number of edges, and num_qubits = number of vertices
def QAOA_proxy(p: int, gamma: np.ndarray, beta: np.ndarray, num_constraints: int, num_qubits: int):
    num_costs = num_constraints + 1
    amplitude_proxies = np.zeros([p + 1, num_costs], dtype=complex)
    init_amplitude = np.sqrt(1 / (1 << num_qubits))
    for i in range(num_costs):
        amplitude_proxies[0][i] = init_amplitude

    for current_depth in range(1, p + 1):
        for cost_1 in range(num_costs):
            amplitude_proxies[current_depth][cost_1] = compute_amplitude_sum(
                amplitude_proxies[current_depth - 1], gamma[current_depth - 1], beta[current_depth - 1], cost_1, num_constraints, num_qubits
            )

    expected_proxy = 0
    for cost in range(num_costs):
        expected_proxy += number_with_cost_proxy(cost, num_constraints, num_qubits) * (abs(amplitude_proxies[p][cost]) ** 2) * cost

    return amplitude_proxies, expected_proxy
