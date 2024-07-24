from numba import njit, jit
import numpy as np
import time
import scipy
import typing
from juliacall import Main as jl
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
jl.seval(f'include("{dir_path}/QAOA_proxy.jl")')

"""
This file implements a version of the QAOA proxy algorithm for MaxCut from:
https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.6.023171

But using direct linear approxmations of distributions
"""

###
### The following functions are defined, but QAOA_proxy now uses the julia
### implementations instead of the python ones. So most of the functions in this
### file will never be called. But they can be called using QAOA_proxy_python
###

# Gives the y-value at x=current_time on the line between (start_time, start_value) and (end_time, end_value)
@njit
def line_between(current_time: float, start_time: float, start_value: float, end_time: float, end_value: float) -> float:
    # Goes from 0 to 1 as current_time goes from start_time to end_time
    relative_time = (current_time - start_time) / (end_time - start_time)

    # Goes from start_value to end_value as relative_time goes from 0 to 1
    return (1 - relative_time) * start_value + relative_time * end_value


#             /\height
#            /  \
# _ _ _ left/    \right _ _ _ given x, returns the corresponding y value on the preceeding curve
@njit
def triangle_value(x: int, left: int, right: int, height: float) -> float:
    return max(0, min(x - left, right - x) * 2 * height / (right - left))


# P(c') from paper but dumber
@njit
def prob_cost(cost: int, num_constraints: int) -> float:
    return 4 / ((num_constraints + 1) ** 2) * min(cost + 1, num_constraints + 1 - cost)


# N(c') from paper but dumber
@njit
def number_with_cost_proxy(cost: int, num_constraints: int, num_qubits: int) -> float:
    scale = 1 << num_qubits
    return prob_cost(cost, num_constraints) * scale


# N(c'; d, c) from paper but instead of a multinomial distribution, we just approximate by a prism whose cross-sections at fixed distances are triangles
# TODO: This only works for prob_edge = 0.5
@njit
def number_of_costs_at_distance_proxy(cost_1: int, cost_2: int, distance: int, num_constraints: int, num_qubits: int) -> float:
    # Want distance to be between 0 and num_qubits//2 since further distance corresponds to being near the bitwise complement (which has the same cost)
    reflected_distance = distance
    if distance > num_qubits // 2:
        reflected_distance = num_qubits - distance

    # Approximate the peak value of the paper's multinomial distribution (roughly)
    h_peak = 1 << (num_qubits - 4)
    # Take the peak height at reflected_distance to be on the straight line between (0 or num_qubits, 1) and (num_qubits/2, h_peak)
    h_at_cost_2 = line_between(reflected_distance, 0, 1, num_qubits / 2, h_peak)
    # Let the peak height at reflected_distance occur where cost_2 is on the stright line between cost_1 and num_constraints/2
    center = line_between(reflected_distance, 0, cost_1, num_qubits / 2, num_constraints / 2)
    left = center - reflected_distance - 1
    right = center + reflected_distance + 1

    return triangle_value(cost_2, left, right, h_at_cost_2)


# Computes the sum inside the for loop of Algorithm 1 in paper using dumb approximations
@njit
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
# Algorithm 1 from paper using dumb approximations
# num_constraints = number of edges, and num_qubits = number of vertices
@njit
def QAOA_proxy_python(p: int, gamma: np.ndarray, beta: np.ndarray, num_constraints: int, num_qubits: int, terms_to_drop_in_expectation: int = 0):
    num_costs = num_constraints + 1
    amplitude_proxies = np.zeros((p + 1, num_costs), dtype=np.complex128) # (p+1, num_costs) needs to be a tuple, not a list, in order to play nicely with numba. Also, dtype must be made more concrete (complex128 instead of complex)
    init_amplitude = np.sqrt(1 / (1 << num_qubits))
    for i in range(num_costs):
        amplitude_proxies[0][i] = init_amplitude

    for current_depth in range(1, p + 1):
        for cost_1 in range(num_costs):
            amplitude_proxies[current_depth][cost_1] = compute_amplitude_sum(
                amplitude_proxies[current_depth - 1], gamma[current_depth - 1], beta[current_depth - 1], cost_1, num_constraints, num_qubits
            )

    expected_proxy = 0
    for cost in range(terms_to_drop_in_expectation, num_costs):
        expected_proxy += number_with_cost_proxy(cost, num_constraints, num_qubits) * (abs(amplitude_proxies[p][cost]) ** 2) * cost

    return amplitude_proxies, expected_proxy

"""
Julia version
"""
def QAOA_proxy(p: int, gamma: np.ndarray, beta: np.ndarray, num_constraints: int, num_qubits: int, terms_to_drop_in_expectation: int = 0):
    return jl.QAOA_proxy(p, gamma, beta, num_constraints, num_qubits, terms_to_drop_in_expectation)


def inverse_proxy_objective_function(num_constraints: int, num_qubits: int, p: int, expectations: list[np.ndarray] | None) -> typing.Callable:
    def inverse_objective(*args) -> float:
        gamma, beta = args[0][:p], args[0][p:]
        _, expectation = QAOA_proxy(p, gamma, beta, num_constraints, num_qubits)
        current_time = time.time()

        if expectations is not None:
            expectations.append((current_time, expectation))

        return -expectation

    return inverse_objective

# Currently only implemented for prob_edge = 0.5
def QAOA_proxy_run(
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
        inverse_proxy_objective_function(num_constraints, num_qubits, p, expectations), init_freq, args=(), method=optimizer_method, options=optimizer_options
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
    _, expectation = QAOA_proxy(p, gamma, beta, num_constraints, num_qubits)

    return {
        "gamma": gamma,
        "beta": beta,
        "expectation": expectation,
        "runtime": end_time - start_time,  # measured in seconds
        "num_QAOA_calls": result.nfev,  # Calls to the proxy of course
        "classical_opt_success": result.success,
        "scipy_opt_message": result.message,
    }

#cost = 1
#num_constraints = 21
#prob_edge = 0.4
#cost_1 = 1
#cost_2 = 2
#distance = 1
#num_qubits = 10
#common_constraints = 3
#prev_amplitudes = np.full(22, 1/1000, dtype=complex)
#gamma = 0.5
#beta = 0.6
#p = 4
#gamma_vec = np.ones(4)
#beta_vec = np.ones(4)
#terms_to_drop_in_expectation = 1
#
#start = time.time()
#print("QAOA_proxy ", QAOA_proxy(p, gamma_vec, beta_vec, num_constraints, num_qubits, terms_to_drop_in_expectation))
#end = time.time()
#print("Elapsed time: ", end-start)
#start = time.time()
#print("QAOA_proxy ", QAOA_proxy(p, gamma_vec, beta_vec, num_constraints, num_qubits, terms_to_drop_in_expectation))
#end = time.time()
#print("Elapsed time: ", end-start)