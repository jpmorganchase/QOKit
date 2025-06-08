import numpy as np
import time
import os
import json
from qokit.qaoa_circuit_portfolio import get_parameterized_qaoa_circuit
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# --- Cost Function for QAOA ---
def qaoa_cost_function(params, po_problem_arg, p_layers, num_shots_simulator, transpilation_level, cost_function_calls,
                       simulator_backend, run_id):
    """
    Computes the QAOA cost function for a given set of parameters (betas and gammas).
    Returns the expected energy and a dictionary of detailed timing components.
    """
    cost_function_calls[0] += 1  # Increment call counter

    gammas = params[:p_layers]
    betas = params[p_layers:]

    # Timings dictionary for this specific call
    call_timings = {}

    # Circuit build
    start_circuit_build = time.perf_counter()
    qaoa_circuit = get_parameterized_qaoa_circuit(
        po_problem=po_problem_arg,
        depth=p_layers,
        gamma=gammas,
        beta=betas
    )
    qaoa_circuit.measure_all()
    call_timings['time_circuit_build'] = time.perf_counter() - start_circuit_build

    # Transpile
    start_transpile = time.perf_counter()
    transpiled_circuit = transpile(qaoa_circuit, simulator_backend, optimization_level=transpilation_level)
    call_timings['time_transpile'] = time.perf_counter() - start_transpile

    # Execute
    start_execute = time.perf_counter()
    job = simulator_backend.run(transpiled_circuit, shots=num_shots_simulator)
    result = job.result()
    counts = result.get_counts(transpiled_circuit)
    call_timings['time_execute'] = time.perf_counter() - start_execute

    # Energy calculation
    start_energy_calc = time.perf_counter()
    expected_energy = 0
    total_shots = sum(counts.values())

    if total_shots == 0:
        print(f"Warning: No shots recorded for bitstring counts in cost function. Counts: {counts}")
        # Return infinite energy and 0 timings if no shots, to avoid division by zero later
        return np.inf, {k: 0.0 for k in call_timings.keys()}  # Return 0 for all timings

    J = po_problem_arg["J"]
    h = po_problem_arg["h"]

    for bitstring, count in counts.items():
        x = np.array([int(b) for b in bitstring[::-1]])

        energy_for_bitstring = 0
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                if (i, j) in J:
                    energy_for_bitstring += J[(i, j)] * x[i] * x[j]
                elif (j, i) in J:
                    energy_for_bitstring += J[(j, i)] * x[i] * x[j]
        for i in range(len(x)):
            if i in h:
                energy_for_bitstring += h[i] * x[i]

        expected_energy += energy_for_bitstring * count

    call_timings['time_energy_calc'] = time.perf_counter() - start_energy_calc

    # --- Print individual component timings (optional, for live monitoring) ---
    if cost_function_calls[0] % 5 == 0 or cost_function_calls[0] == 1:
        print(f"Run {run_id}, Cost Func Call {cost_function_calls[0]}: "
              f"Build: {call_timings['time_circuit_build']:.4f}s, "
              f"Transpile: {call_timings['time_transpile']:.4f}s, "
              f"Execute: {call_timings['time_execute']:.4f}s, "
              f"Energy Calc: {call_timings['time_energy_calc']:.4f}s")

    return expected_energy / total_shots, call_timings


# --- Worker Function for Parallel Processing ---
def run_single_optimization(initial_point_tuple, po_problem_arg, p_layers, max_iterations_optimizer,
                            num_shots_simulator, run_id, transpilation_level):
    """
    Performs a single QAOA optimization run from a given initial point.
    Collects and returns average timings from qaoa_cost_function calls.
    """
    start_time = time.perf_counter()
    initial_point = np.array(initial_point_tuple)

    simulator = AerSimulator()

    print(
        f"Run {run_id}: Starting optimization from initial point: {np.round(initial_point, 3)} with N={po_problem_arg['N']}, p={p_layers}, MaxIter={max_iterations_optimizer}, TL={transpilation_level}")

    cost_function_calls = [0]  # Mutable list to pass count by reference

    # Lists to store timings from each cost function evaluation
    all_circuit_build_times = []
    all_transpile_times = []
    all_execute_times = []
    all_energy_calc_times = []

    # Wrapper function to capture timings and pass to minimize
    def cost_function_wrapper(params):
        energy, timings = qaoa_cost_function(params, po_problem_arg, p_layers, num_shots_simulator,
                                             transpilation_level, cost_function_calls, simulator, run_id)

        # Store timings from this call
        all_circuit_build_times.append(timings.get('time_circuit_build', 0.0))
        all_transpile_times.append(timings.get('time_transpile', 0.0))
        all_execute_times.append(timings.get('time_execute', 0.0))
        all_energy_calc_times.append(timings.get('time_energy_calc', 0.0))

        return energy

    try:
        bounds = [(0, 2 * np.pi)] * p_layers + [(0, np.pi)] * p_layers

        result = minimize(cost_function_wrapper, initial_point,  # Use the wrapper here
                          method='COBYLA', bounds=bounds,
                          options={'maxiter': max_iterations_optimizer,
                                   'disp': False})

        end_time = time.perf_counter()
        runtime = end_time - start_time

        print(f"Run {run_id}: Optimization completed. NFEV: {result.nfev}, Runtime: {runtime:.2f}s")

        # Calculate average timings per cost function call for this optimization run
        avg_circuit_build_time = np.mean(all_circuit_build_times) if all_circuit_build_times else 0.0
        avg_transpile_time = np.mean(all_transpile_times) if all_transpile_times else 0.0
        avg_execute_time = np.mean(all_execute_times) if all_execute_times else 0.0
        avg_energy_calc_time = np.mean(all_energy_calc_times) if all_energy_calc_times else 0.0

        return {
            "run_id": run_id,
            "optimal_params": result.x.tolist(),
            "optimal_energy": result.fun,
            "nfev": result.nfev,
            "success": result.success,
            "message": result.message,
            "runtime_seconds": runtime,  # Total runtime for this single optimization process
            "avg_circuit_build_time_per_call": avg_circuit_build_time,
            "avg_transpile_time_per_call": avg_transpile_time,
            "avg_execute_time_per_call": avg_execute_time,
            "avg_energy_calc_time_per_call": avg_energy_calc_time
        }

    except Exception as e:
        end_time = time.perf_counter()
        runtime = end_time - start_time
        print(f"Run {run_id}: Optimization failed - {e}")
        # In case of failure, calculate averages from collected data, or return 0 if no calls
        avg_circuit_build_time = np.mean(all_circuit_build_times) if all_circuit_build_times else 0.0
        avg_transpile_time = np.mean(all_transpile_times) if all_transpile_times else 0.0
        avg_execute_time = np.mean(all_execute_times) if all_execute_times else 0.0
        avg_energy_calc_time = np.mean(all_energy_calc_times) if all_energy_calc_times else 0.0

        return {
            "run_id": run_id,
            "optimal_energy": float('inf'),
            "optimal_params": None,
            "nfev": cost_function_calls[0],
            "success": False,
            "message": str(e),
            "runtime_seconds": runtime,
            "avg_circuit_build_time_per_call": avg_circuit_build_time,
            "avg_transpile_time_per_call": avg_transpile_time,
            "avg_execute_time_per_call": avg_execute_time,
            "avg_energy_calc_time_per_call": avg_energy_calc_time
        }