from __future__ import annotations
import numpy as np
from .utils import precompute_energies, reverse_array_index_bit_order
from .portfolio_optimization import get_configuration_cost_kw, po_obj_func, portfolio_brute_force
from qokit.qaoa_circuit_portfolio import generate_dicke_state_fast, get_parameterized_qaoa_circuit
from .qaoa_objective import get_qaoa_objective
from qiskit.circuit import ParameterVector # <--- ADD THIS LINE!
from qiskit.primitives import Estimator
from qiskit.quantum_info import Pauli, SparsePauliOp


from qiskit_aer import Aer # Assuming you're using Aer, this should be here
from qiskit.providers import Backend # <--- ADD THIS LINE!
# from qiskit.primitives import Estimator # Add if qokit uses Estimator and it's not already imported
# from qiskit.quantum_info import Pauli, SparsePauliOp # Add if qokit uses these and they're not already imported

# ... (rest of your imports in qaoa_objective_portfolio.py) ...

from qokit.qaoa_circuit_portfolio import get_parameterized_qaoa_circuit
# ... (rest of the file content) ...
# Original conceptual signature might look something like this:
# def get_qaoa_portfolio_objective(po_problem, p=1, ini='uniform', mixer='trotter_ring', T=1, simulator='python'):

# --- MODIFIED CODE FOR qaoa_objective_portfolio.py ---
from qokit.qaoa_circuit_portfolio import get_parameterized_qaoa_circuit # Ensure this import exists
# ... other imports ...

def get_qaoa_portfolio_objective(po_problem, p=1, ini='uniform', mixer='trotter_ring', T=1, simulator='python', mixer_topology='complete'): # <<-- ADD mixer_topology here with a default
    """
    Returns a function that evaluates the QAOA objective for portfolio optimization.

    Args:
        po_problem (dict): Dictionary defining the portfolio optimization problem (N, K, J, h).
        p (int): Number of QAOA layers.
        ini (str): Initial state type ('uniform' or 'dicke').
        mixer (str): Mixer Hamiltonian type ('trotter_ring' or 'x_mixer').
        T (float): Trotterization step (usually 1).
        simulator (Union[str, Backend]): Simulator to use ('python', 'qiskit', or a Qiskit Backend object).
        mixer_topology (str): Defines the connectivity for XY interactions in the mixer.
                              'complete', 'linear', or 'ring'. (New argument)
    Returns:
        Callable: A function that takes QAOA parameters (gamma, beta) and returns the objective value.
    """
    N = po_problem["N"]
    K = po_problem["K"]
    J = po_problem["J"]
    h = po_problem["h"]
    gamma = ParameterVector('gamma', p) # p is the number of QAOA layers
    beta = ParameterVector('beta', p)


    if simulator == 'python':
        # ... (existing python simulator logic) ...
        # This part typically precomputes energies for all 2^N bitstrings
        # and would not use the qiskit circuit generation directly.
        # So, mixer_topology might not directly apply here unless it affects the energy precomputation,
        # which it doesn't. This path is for small N only.
        # For N > 20, we assume simulator is a qiskit backend.
        pass

    
    elif simulator == 'qiskit' or isinstance(simulator, Backend): # Assuming Backend is imported from qiskit.providers
        # get_parameterized_qaoa_circuit creates the Qiskit circuit
        parameterized_circuit = get_parameterized_qaoa_circuit(
            po_problem=po_problem,
            depth=p,
            ini_type=ini,
            mixer_type=mixer,
            T=T,
            simulator='qiskit',
            mixer_topology=mixer_topology,
            gamma=gamma, # Pass gamma
            beta=beta      # Pass beta
        )


        # ... (rest of the qiskit execution logic using Estimator or custom energy calculation) ...
        # This part should be consistent with how qokit handles qiskit backends.
        # Assuming qokit uses a Qiskit Estimator for execution:
        if isinstance(simulator, Backend): # If a Backend object is passed
            qiskit_backend = simulator
        else: # If 'qiskit' string is passed
            qiskit_backend = Aer.get_backend('aer_simulator') # Default Qiskit Aer simulator

        # This is a conceptual representation of how qokit would then evaluate the circuit
        # It might use Estimator or a custom execution loop
        def qaoa_objective_function(params):
            # Assign parameters to the circuit
            bound_circuit = parameterized_circuit.assign_parameters(params)
            # Execute the circuit. The 'shots' are set on the qiskit_backend object itself.
            # This is where the simulation happens.
            # You might need to adjust this part if qokit has a specific Estimator wrapper.
            # For now, let's assume qokit's internal mechanism handles the Backend object correctly.
            # Initialize Estimator without the 'backend' argument
            estimator = Estimator()
            
            # Get the shots from the backend's options that were set earlier.
            # This assumes qiskit_backend is a Qiskit Backend object (like AerSimulator)
            # and that you set its shots using simulator_backend.set_options(shots=shots)
            execution_shots = qiskit_backend.options.get('shots', None) # Get shots from the backend's options
            
            # Convert J and h to Pauli strings for the Estimator
            # (This part should be existing in your file, ensuring N is accessible here)
            # For example, N would be available from po_problem in the outer function's scope.
            N = po_problem["N"] # Make sure N is available in this scope
            
            pauli_list = []
            # Add quadratic terms (J_ij Z_i Z_j)
            for (i, j), val in J.items(): # J and h are defined in the outer get_qaoa_portfolio_objective function
                if val != 0:
                    z_op = ['I'] * N
                    z_op[i] = 'Z'
                    z_op[j] = 'Z'
                    # pauli_list.append((Pauli(''.join(z_op)), val))
                    pauli_list.append((''.join(z_op), val)) # <<-- CORRECTED: Pass the string directly

            
            # Add linear terms (h_i Z_i)
            for i, val in h.items():
                if val != 0:
                    z_op = ['I'] * N
                    z_op[i] = 'Z'
                    pauli_list.append((''.join(z_op), val)) # <<-- CORRECTED: Pass the string directly
                    # pauli_list.append((Pauli(''.join(z_op)), val))
            
            # Convert to an observable for the Estimator
            observable = SparsePauliOp.from_list(pauli_list)
            
            # Execute and get expectation value
            # Pass both the bound circuit and the observable as lists, and specify the backend and shots here.
            job = estimator.run(
                circuits=[bound_circuit], # Pass the circuit as a list
                observables=[observable], # Pass the observable as a list
                backend=qiskit_backend, # <<-- Pass the backend here!
                shots=execution_shots # <<-- Pass the shots here!
            )
            result = job.result()
            expectation_value = result.values[0]

            # Example of what qokit might do internally (Estimator approach):
            # from qiskit.primitives import Estimator
            return expectation_value.real

        return qaoa_objective_function

    else:
        raise ValueError(f"Unknown simulator type: {simulator}")