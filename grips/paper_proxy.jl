
#import numpy as np
#import math
#import typing
#import time
#import scipy
#from scipy.stats import binom, multinomial
using Distributions
using TimerOutputs

#"""
#This file implements the QAOA proxy algorithm for MaxCut from:
#https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.6.023171
#"""



"""
P(c') from paper
"""
function prob_cost_paper(cost::Int, num_constraints::Int, prob_edge::Real=0.5)::Float64

    binomial_distribution = Binomial(num_constraints, prob_edge)
    return pdf(binomial_distribution, cost)
end
#def prob_cost_paper(cost: int, num_constraints: int, prob_edge: float = 0.5) -> float:
#    return binom.pmf(cost, num_constraints, prob_edge)


"""
N(c') from paper
"""
function number_with_cost_paper_proxy(cost::Int, num_constraints::Int, num_qubits::Int, 
        prob_edge::Float64 = 0.5)::Float64

    scale = 1 << num_qubits
    return prob_cost_paper(cost, num_constraints, prob_edge) * scale
end


"""
P(b, c'-b, c-b | d) from paper
"""
function prob_common_at_distance_paper(num_constraints::Int, num_qubits::Int, common_constraints::Int,
        cost_1::Int, cost_2::Int, distance::Int)::Float64

    #prob_same = (math.comb(num_constraints - distance, 2) + math.comb(distance, 2)) / math.comb(num_constraints, 2)
    prob_same = (binomial(num_qubits - distance, 2) + binomial(distance, 2)) / binomial(num_qubits, 2)
    prob_neither = prob_same / 2
    prob_both = prob_neither

    prob_one = (1 - prob_neither - prob_both) / 2
    probability_vec = [prob_both, prob_one, prob_one, prob_neither]
    multinomial_distribution = Multinomial(num_constraints, probability_vec)
    k_vec = [common_constraints, cost_1 - common_constraints, cost_2 - common_constraints, num_constraints + common_constraints - (cost_1 + cost_2)]
    return pdf(multinomial_distribution, k_vec)
end

"""
N(c'; d, c) from paper
"""
function number_of_costs_at_distance_paper_proxy(cost_1::Int, cost_2::Int, distance::Int, 
        num_constraints::Int, num_qubits::Int, prob_edge::Float64 = 0.5)::Float64
    sum = 0
    start_index = max(0, cost_1 + cost_2 - num_constraints)
    end_index = min(cost_1, cost_2)
    for common_constraints in start_index:end_index
        sum += prob_common_at_distance_paper(num_constraints, num_qubits, common_constraints, cost_1, cost_2, distance)
    end

    p_cost = prob_cost_paper(cost_1, num_constraints, prob_edge)
    return (binomial(num_qubits, distance) / p_cost) * sum
end

"""
Computes the sum inside the for-loop of Algorithm 1 in the paper
"""
function compute_amplitude_sum_paper(prev_amplitudes::AbstractVector{ComplexF64}, gamma::Float64, beta::Float64, cost_1::Int, num_constraints::Int, num_qubits::Int)::ComplexF64
    sum::ComplexF64 = 0.0+0.0im
    for cost_2 in 0:num_constraints
        for distance in 0:num_qubits
            # Should I np-ify all of the stuff here?
            beta_factor = (cos(beta) ^ (num_qubits - distance)) * ((-1im * sin(beta)) ^ distance)
            gamma_factor = exp(-1im * gamma * cost_2)
            num_costs_at_distance = number_of_costs_at_distance_paper_proxy(cost_1, cost_2, distance, num_constraints, num_qubits)
            sum += beta_factor * gamma_factor * prev_amplitudes[1+cost_2] * num_costs_at_distance
        end
    end
    return sum
end

"""
TODO: What if instead of optimizing expectation proxy we instead optimize high cost amplitudes (using e.g. exponential weighting)
Algorithm 1 from paper
num_constraints = number of edges, and num_qubits = number of vertices
"""
function QAOA_paper_proxy(p::Int, gamma::AbstractVector{Float64}, beta::AbstractVector{Float64},
        num_constraints::Int, num_qubits::Int, terms_to_drop_in_expectation::Int = 0)

    num_costs = num_constraints + 1
    amplitude_proxies = zeros(ComplexF64, p+1, num_costs)
    init_amplitude = sqrt(1 / (1 << num_qubits))
    amplitude_proxies[1,:] .= init_amplitude # Memory inefficient array access

    for current_depth in 1:p
        for cost_1 in 0:num_costs-1
            amplitude_proxies[1+current_depth, 1+cost_1] = compute_amplitude_sum_paper(
                amplitude_proxies[current_depth,:], gamma[current_depth], beta[current_depth], cost_1, num_constraints, num_qubits
            )
        end
    end
    println(amplitude_proxies[:,1])

    expected_proxy = 0
    for cost in terms_to_drop_in_expectation:num_costs-1
        expected_proxy += number_with_cost_paper_proxy(cost, num_constraints, num_qubits) * (abs(amplitude_proxies[end, 1+cost]) ^ 2) * cost
    end

    return amplitude_proxies, expected_proxy
end


cost = 1
num_constraints = 21
prob_edge = 0.4
cost_1 = 1
cost_2 = 2
distance = 1
num_qubits = 10
common_constraints = 3
prev_amplitudes = ones(ComplexF64, 22) * 1/1000
gamma = 0.5
beta = 0.6
p = 4
gamma_vec = ones(4)
beta_vec = ones(4)
terms_to_drop_in_expectation = 1

to = TimerOutput()
println("prob_cost_paper ", prob_cost_paper(cost, num_constraints, prob_edge))
println("number_with_cost_paper_proxy ", number_with_cost_paper_proxy(cost, num_constraints, num_qubits, prob_edge))
println("prob_common_at_distance_paper ", prob_common_at_distance_paper(num_constraints, num_qubits, common_constraints, cost_1, cost_2, distance))
println("number_of_costs_at_distance_paper_proxy ", number_of_costs_at_distance_paper_proxy(cost_1, cost_2, distance, num_constraints, num_qubits, prob_edge))
println("compute_amplitude_sum_paper ", compute_amplitude_sum_paper(prev_amplitudes, gamma, beta, cost_1, num_constraints, num_qubits))
#QAOA_paper_proxy(p, gamma_vec, beta_vec, num_constraints, num_qubits, terms_to_drop_in_expectation)
println("QAOA_paper_proxy ", QAOA_paper_proxy(p, gamma_vec, beta_vec, num_constraints, num_qubits, terms_to_drop_in_expectation))
@timeit to "Finished" println("Finished")
show(to)

#=
function inverse_paper_proxy_objective_function(num_constraints::Int, num_qubits::Int, p::Int, expectations::Union{AbstractVector, Missing}=missing)::Function
    function inverse_objective(args...)::Float64
        gamma, beta = args[1][1:p], args[2][1+p:end]
        _, expectation = QAOA_paper_proxy(p, gamma, beta, num_constraints, num_qubits)
        current_time = time.time()

        if !ismissing(expectations)
            push!(expectations, (current_time, expectation))
        end

        return -expectation
    end

    return inverse_objective
end


function QAOA_paper_proxy_run(
    num_constraints::Int,
    num_qubits::Int,
    p::Int,
    init_gamma::AbstractVector{Real},
    init_beta::AbstractVector{Real},
@@@    expectations::Union{, Missing} list[np.ndarray] | None = None,
)
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
=#
