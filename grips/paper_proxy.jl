
#import numpy as np
#import math
#import typing
#import time
#import scipy
#from scipy.stats import binom, multinomial
using Distributions
using TimerOutputs
using PythonCall
using BenchmarkTools

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

gamma and beta are explicitly-typed to force explicit conversion of numpy
arrays, which I found gives ~20% speedup.
"""
function QAOA_paper_proxy(p::Int, gamma::Vector{Float64}, beta::Vector{Float64},
        num_constraints::Int, num_qubits::Int, terms_to_drop_in_expectation::Int = 0)
    println("Hello world.")
    println(typeof(gamma))
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

"""
Convert numpy arrays to julia arrays before doing QAOA_proxy. This gives about
~20% speedup over using numpy arrays in julia.
"""
function QAOA_paper_proxy(p, gamma::PyArray, beta::PyArray, num_constraints, num_qubits, terms_to_drop_in_expectation)
    t = @elapsed pyconvert(Vector, gamma)
    println("pyconvert time = ", t)
    return QAOA_paper_proxy(
        pyconvert(Int, p),
        pyconvert(Vector, gamma),
        pyconvert(Vector, beta),
        pyconvert(Int, num_constraints),
        pyconvert(Int, num_qubits),
        pyconvert(Int, terms_to_drop_in_expectation),
    )
end