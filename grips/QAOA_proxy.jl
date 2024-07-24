"""
Gives the y-value at x=current_time on the line between (start_time, start_value) and (end_time, end_value)
"""
function line_between(current_time::Real, start_time::Real,
        start_value::Real, end_time::Real,
        end_value::Real)::Float64

    # Goes from 0 to 1 as current_time goes from start_time to end_time
    relative_time = (current_time - start_time) / (end_time - start_time)

    # Goes from start_value to end_value as relative_time goes from 0 to 1
    return (1 - relative_time) * start_value + relative_time * end_value
end

"""
             /\\height
            /  \
 _ _ _ left/    \\right _ _ _ given x, returns the corresponding y value on the preceeding curve
"""
function triangle_value(x::Int, left::Real, right::Real, height::Real)::Float64
    return max(0, min(x - left, right - x) * 2 * height / (right - left))
end

"""
P(c') from paper but dumber
"""
function prob_cost(cost::Int, num_constraints::Int)::Float64
    return 4 / ((num_constraints + 1) ^ 2) * min(cost + 1, num_constraints + 1 - cost)
end

"""
N(c') from paper but dumber
"""
function number_with_cost_proxy(cost::Int, num_constraints::Int,
        num_qubits::Int)::Float64
    scale = 1 << num_qubits
    return prob_cost(cost, num_constraints) * scale
end

"""
N(c'; d, c) from paper but instead of a multinomial distribution, we just approximate by a prism whose cross-sections at fixed distances are triangles
TODO: This only works for prob_edge = 0.5
"""
function number_of_costs_at_distance_proxy(cost_1::Int, cost_2::Int,
        distance::Int, num_constraints::Int, num_qubits::Int)::Float64

    # Want distance to be between 0 and num_qubits//2 since further distance corresponds to being near the bitwise complement (which has the same cost)
    reflected_distance = distance
    if distance > div(num_qubits, 2)
        reflected_distance = num_qubits - distance
    end

    # Approximate the peak value of the paper's multinomial distribution (roughly)
    h_peak = 1 << (num_qubits - 4)
    # Take the peak height at reflected_distance to be on the straight line between (0 or num_qubits, 1) and (num_qubits/2, h_peak)
    h_at_cost_2 = line_between(reflected_distance, 0, 1, num_qubits / 2, h_peak)
    # Let the peak height at reflected_distance occur where cost_2 is on the stright line between cost_1 and num_constraints/2
    center = line_between(reflected_distance, 0, cost_1, num_qubits / 2, num_constraints / 2)
    left = center - reflected_distance - 1
    right = center + reflected_distance + 1

    return triangle_value(cost_2, left, right, h_at_cost_2)
end

"""
Computes the sum inside the for loop of Algorithm 1 in paper using dumb approximations
"""
function compute_amplitude_sum(prev_amplitudes::AbstractVector{ComplexF64},
        gamma::Real, beta::Real, cost_1::Int, num_constraints::Int,
        num_qubits::Int)::ComplexF64

    sum = 0
    for cost_2 in 0:num_constraints
        for distance in 0:num_qubits
            # Should I np-ify all of the stuff here?
            beta_factor = (cos(beta) ^ (num_qubits - distance)) * ((-1im * sin(beta)) ^ distance)
            gamma_factor = exp(-1im * gamma * cost_2)
            num_costs_at_distance = number_of_costs_at_distance_proxy(cost_1, cost_2, distance, num_constraints, num_qubits)
            sum += beta_factor * gamma_factor * prev_amplitudes[1+cost_2] * num_costs_at_distance
        end
    end
    return sum
end

"""
Currently only implemented for prob_edge = 0.5
"""
function QAOA_proxy(p::Int, gamma::Vector{Float64}, beta::Vector{Float64}, num_constraints::Int, num_qubits::Int, terms_to_drop_in_expectation::Int = 0)
    num_costs = num_constraints + 1
    amplitude_proxies = zeros(ComplexF64, p + 1, num_costs)
    init_amplitude = sqrt(1 / (1 << num_qubits))
    amplitude_proxies[1,:] .= init_amplitude # Memory inefficient, would be better to fill a column than a row
    
    for current_depth in 1:p
        for cost_1 in 0:num_costs-1
            amplitude_proxies[1+current_depth,1+cost_1] = compute_amplitude_sum(
                amplitude_proxies[current_depth,:], gamma[current_depth], beta[current_depth], cost_1, num_constraints, num_qubits
            )
        end
    end

    expected_proxy = 0
    for cost in terms_to_drop_in_expectation:num_costs-1
        expected_proxy += number_with_cost_proxy(cost, num_constraints, num_qubits) * (abs(amplitude_proxies[end,1+cost]) ^ 2) * cost
    end

    return amplitude_proxies, expected_proxy
end

"""
Convert numpy arrays to julia arrays before doing QAOA_proxy.
(should check whether this impacts performance)
"""
function QAOA_proxy(p, gamma::PyArray, beta::PyArray, num_constraints, num_qubits, terms_to_drop_in_expectation)
    return QAOA_proxy(
        pyconvert(Int, p),
        pyconvert(Vector, gamma),
        pyconvert(Vector, beta),
        pyconvert(Int, num_constraints),
        pyconvert(Int, num_qubits),
        pyconvert(Int, terms_to_drop_in_expectation),
    )
end

#=
using BenchmarkTools

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

# Should really run all functions here once to make sure they are compiled.

@time println("QAOA_proxy ", QAOA_proxy(p, gamma_vec, beta_vec, num_constraints, num_qubits, terms_to_drop_in_expectation))
@btime QAOA_proxy(p, gamma_vec, beta_vec, num_constraints, num_qubits, terms_to_drop_in_expectation)
=#