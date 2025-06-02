import numpy as np
import time
import os
import json
from qokit.qaoa_circuit_portfolio import get_parameterized_qaoa_circuit
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile

# --- Qiskit Simulator Setup ---
simulator = None
try:
    # Try importing AerSimulator directly from qiskit_aer (modern approach)
    from qiskit_aer import AerSimulator

    simulator = AerSimulator()
    print("Using AerSimulator from qiskit_aer")
except ImportError:
    try:
        # Fallback for older Qiskit versions or specific environments
        # where Aer is in qiskit.providers.aer
        from qiskit.providers.aer import Aer

        simulator = Aer.get_backend('qasm_simulator')
        print("Using Aer.get_backend('qasm_simulator') from qiskit.providers.aer")
    except ImportError:
        # If neither works, then qiskit-aer is likely not installed or accessible
        print("Could not import AerSimulator or Aer from qiskit_aer or qiskit.providers.aer.")
        print(
            "Please ensure 'qiskit-aer' is installed (pip install qiskit-aer) or that your Qiskit installation is complete.")
        # Re-raise the error to stop execution, as the simulator is critical
        raise ImportError("Qiskit Aer backend not found. Please install qiskit-aer or check your Qiskit installation.")

if simulator is None:
    raise RuntimeError("Simulator backend could not be initialized. Check Qiskit Aer installation.")


# --- Cost Function for QAOA ---
def qaoa_cost_function(params, po_problem_arg, p_layers, num_shots_simulator, cost_function_calls):
    """
    Computes the QAOA cost function for a given set of parameters (betas and gammas).
    This function will be minimized by scipy.optimize.
    """
    cost_function_calls[0] += 1  # Increment call counter

    # Extract gammas and betas from the params array
    gammas = params[:p_layers]
    betas = params[p_layers:]

    # Construct the QAOA circuit
    qaoa_circuit = get_parameterized_qaoa_circuit(
        po_problem=po_problem_arg,
        depth=p_layers,
        gamma=gammas,
        beta=betas
    )
    qaoa_circuit.measure_all()  # Ensure measurements are added

    # Transpile for the simulator
    transpiled_circuit = transpile(qaoa_circuit, simulator, optimization_level=0)

    # Execute the circuit on the simulator
    job = simulator.run(transpiled_circuit, shots=num_shots_simulator)  # Uses the passed num_shots_simulator
    result = job.result()
    counts = result.get_counts(transpiled_circuit)

    # Calculate the expectation value (cost)
    expected_energy = 0
    total_shots = sum(counts.values())

    # Handle the case where total_shots is 0 (shouldn't happen with num_shots_simulator > 0 and no error)
    if total_shots == 0:
        # This is an error condition, return a very high energy to penalize
        print(f"Warning: No shots recorded for bitstring counts in cost function. Counts: {counts}")
        return np.inf

    J = po_problem_arg["J"]
    h = po_problem_arg["h"]

    for bitstring, count in counts.items():
        x = np.array([int(b) for b in
                      bitstring[::-1]])  # Convert bitstring to numpy array (reversed to match Qiskit qubit order)

        # Calculate energy for this bitstring
        energy_for_bitstring = 0

        # Quadratic terms J_ij x_i x_j
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                if (i, j) in J:
                    energy_for_bitstring += J[(i, j)] * x[i] * x[j]
                elif (j, i) in J:
                    energy_for_bitstring += J[(j, i)] * x[i] * x[j]

        # Linear terms h_i x_i
        for i in range(len(x)):
            if i in h:
                energy_for_bitstring += h[i] * x[i]

        expected_energy += energy_for_bitstring * count

    return expected_energy / total_shots


# --- Worker Function for Parallel Processing ---
def run_single_optimization(initial_point_tuple, po_problem_arg, p_layers, max_iterations_optimizer,
                            num_shots_simulator, run_id):
    """
    Performs a single QAOA optimization run from a given initial point.
    This function is designed to be run in parallel processes.
    """
    start_time = time.perf_counter()
    initial_point = np.array(initial_point_tuple)

    print(f"Run {run_id}: Starting optimization from initial point: {np.round(initial_point, 3)}")
    print(f"Run {run_id}: Keys in po_problem_arg: {po_problem_arg.keys()}")

    cost_function_calls = [0]

    try:
        bounds = [(0, 2 * np.pi)] * p_layers + [(0, np.pi)] * p_layers

        # --- DEBUG PRINT STATEMENT UPDATED TO REFLECT VARIABLE ---
        print(
            f"Run {run_id}: DEBUG - COBYLA optimizer options set to: {{'maxiter': {max_iterations_optimizer}, 'disp': False}}")
        # --- END DEBUG PRINT ---

        result = minimize(qaoa_cost_function, initial_point,
                          args=(po_problem_arg, p_layers, num_shots_simulator, cost_function_calls),
                          method='COBYLA', bounds=bounds,
                          options={'maxiter': max_iterations_optimizer,
                                   'disp': False})  # Uses the passed max_iterations_optimizer

        end_time = time.perf_counter()
        runtime = end_time - start_time

        return {
            "run_id": run_id,
            "optimal_params": result.x.tolist(),
            "optimal_energy": result.fun,
            "nfev": result.nfev,
            "success": result.success,
            "message": result.message,
            "runtime_seconds": runtime
        }

    except Exception as e:
        end_time = time.perf_counter()
        runtime = end_time - start_time
        print(f"Run {run_id}: Optimization failed - {e}")
        return {
            "run_id": run_id,
            "optimal_energy": float('inf'),  # Use inf for failed runs to sort correctly
            "optimal_params": None,
            "nfev": cost_function_calls[0],
            "success": False,
            "message": str(e),
            "runtime_seconds": runtime
        }