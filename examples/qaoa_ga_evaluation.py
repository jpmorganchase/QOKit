import numpy as np
from qiskit.primitives import Sampler
from qiskit.circuit import ParameterVector
# We'll use get_parameterized_qaoa_circuit to build the circuit
from qokit.qaoa_circuit_portfolio import get_parameterized_qaoa_circuit

def evaluate_qaoa_fitness(individual, po_problem_arg, p_layers):
    """
    Evaluates the QAOA objective for a given set of parameters (individual) for Genetic Algorithms.
    This function now directly builds the QAOA circuit, runs it on a Sampler, and calculates the
    expectation value from the measurement outcomes. This approach is more robust to qokit
    versioning issues with the get_qaoa_portfolio_objective function.

    Args:
        individual (list): A list of QAOA parameters (betas and gammas).
        po_problem_arg (dict): The portfolio optimization problem definition dictionary
                                (containing J, h, N, etc.).
        p_layers (int): The number of QAOA layers (p).

    Returns:
        tuple: A tuple containing the real part of the estimated energy.
               DEAP expects a tuple for fitness values.
    """
    # Extract gammas and betas from the individual (2*p parameters)
    gammas = individual[:p_layers]
    betas = individual[p_layers:]

    # Define Qiskit ParameterVectors for the circuit construction
    gamma_params = ParameterVector('gamma', p_layers)
    beta_params = ParameterVector('beta', p_layers)

    # 1. Build the parameterized QAOA circuit using qokit's circuit builder
    # We pass the ParameterVectors, which will be assigned the numerical values later.
    qaoa_circuit = get_parameterized_qaoa_circuit(
        po_problem=po_problem_arg,
        depth=p_layers,
        ini_type='dicke',
        mixer_type='trotter_ring',
        T=1,
        simulator=None, # Not directly using simulator for building, Sampler handles execution
        mixer_topology='linear',
        gamma=gamma_params,
        beta=beta_params
    )

    # 2. Assign the numerical parameter values from the 'individual' to the circuit
    bound_circuit = qaoa_circuit.assign_parameters({
        gamma_params: gammas,
        beta_params: betas
    })

    # 3. Add measurements to the circuit for shot-based sampling
    bound_circuit.measure_all()

    # 4. Use a Sampler to run the circuit and get measurement outcomes
    shots_per_evaluation = 1024 # Adjust for desired accuracy vs. speed (e.g., 2048, 4096)
    sampler_instance = Sampler(options={'shots': shots_per_evaluation})

    job = sampler_instance.run(bound_circuit, shots=shots_per_evaluation)
    result = job.result()
    # The result contains quasi-distributions, which are dictionaries of outcome
    # counts/probabilities. We expect one result for one circuit.
    quasi_dists = result.quasi_dists[0]

    # 5. Manually calculate the expectation value (energy) from the probabilities
    total_estimated_energy = 0.0
    N_assets = po_problem_arg["N"]
    J_coeffs = po_problem_arg["J"]
    h_coeffs = po_problem_arg["h"]

    for bitstring_int, probability in quasi_dists.items():
        # Convert integer bitstring to a numpy array of 0s and 1s
        # format(int, '0{N}b') ensures correct binary length with leading zeros
        bitstring_str = format(bitstring_int, f'0{N_assets}b')
        x_vector = np.array([int(bit) for bit in bitstring_str])

        # Calculate the energy for this specific bitstring using the problem's QUBO coefficients
        current_bitstring_energy = 0.0

        # Add linear terms (h)
        for i in range(N_assets):
            # Use .get(i, 0) in case a coefficient for 'i' is missing, defaulting to 0
            current_bitstring_energy += h_coeffs.get(i, 0) * x_vector[i]

        # Add quadratic terms (J)
        for (i, j), J_val in J_coeffs.items():
            current_bitstring_energy += J_val * x_vector[i] * x_vector[j]

        # Accumulate the weighted energy based on the bitstring's probability
        total_estimated_energy += current_bitstring_energy * probability

    # DEAP expects fitness values to be returned as a tuple.
    return (total_estimated_energy,)