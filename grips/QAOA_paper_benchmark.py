
cost = 1
num_constraints = 21
prob_edge = 0.4
cost_1 = 1
cost_2 = 2
distance = 1
num_qubits = 10
common_constraints = 3
prev_amplitudes = np.full(22, 1/1000, dtype=complex)
gamma = 0.5
beta = 0.6
p = 4
gamma_vec = np.ones(4)
beta_vec = np.ones(4)
terms_to_drop_in_expectation = 1

start = time.time()

print("prob_cost_paper ", prob_cost_paper(cost, num_constraints, prob_edge))
print("number_with_cost_paper_proxy ", number_with_cost_paper_proxy(cost, num_constraints, num_qubits, prob_edge))
print("prob_common_at_distance_paper ", prob_common_at_distance_paper(num_constraints, common_constraints, cost_1, cost_2, distance))
print("number_of_costs_at_distance_paper_proxy ", number_of_costs_at_distance_paper_proxy(cost_1, cost_2, distance, num_constraints, num_qubits, prob_edge))
print("compute_amplitude_sum_paper ", compute_amplitude_sum_paper(prev_amplitudes, gamma, beta, cost_1, num_constraints, num_qubits))
#QAOA_paper_proxy(p, gamma_vec, beta_vec, num_constraints, num_qubits, terms_to_drop_in_expectation)
print("QAOA_paper_proxy ", QAOA_paper_proxy(p, gamma_vec, beta_vec, num_constraints, num_qubits, terms_to_drop_in_expectation))

end = time.time()
print("Elapsed time: ", end-start)
