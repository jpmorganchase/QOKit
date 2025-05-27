import numpy as np
from scipy.optimize import minimize
from qiskit import transpile
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.primitives import Sampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA, SPSA, L_BFGS_B
#from qiskit.utils import QuantumInstance # For older Qiskit versions, if needed
from qiskit_aer import AerSimulator # For statevector simulation


#from qokit.portfolio_optimization import get_problem, portfolio_brute_force, get_sk_ini
#from qokit.qaoa_objective_portfolio import get_qaoa_portfolio_objective
#from qokit.qaoa_circuit_portfolio import get_qaoa_circuit
#from qokit.utils import reverse_array_index_bit_order # Utility for statevector handling

# Suppress deprecation warnings for cleaner output in this example
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class DWEEncoder:
    """
    A utility class for Domain-Wall Encoding (DWE) of a single integer variable.
    """
    def __init__(self, max_quantity: int):
        """
        Initializes the DWE encoder for a variable that can take values from 0 to max_quantity.
        Args:
            max_quantity (int): The maximum integer value the variable can take.
                                (e.g., if max_quantity=2, values are 0, 1, 2).
        """
        self.max_quantity = max_quantity
        self.num_qubits = max_quantity # For m values (0 to m-1), DWE uses m-1 qubits.

    def encode_quantity(self, quantity: int) -> str:
        """
        Encodes an integer quantity into its DWE binary string representation.
        Args:
            quantity (int): The integer quantity to encode (0 <= quantity <= self.max_quantity).
        Returns:
            str: The DWE binary string (e.g., "00", "10", "11").
        Raises:
            ValueError: If the quantity is out of the valid range.
        """
        if not (0 <= quantity <= self.max_quantity):
            raise ValueError(f"Quantity {quantity} out of valid range [0, {self.max_quantity}]")

        if quantity == 0:
            return "0" * self.num_qubits
        else:
            return "1" * quantity + "0" * (self.num_qubits - quantity)

    def decode_bitstring(self, bitstring: str) -> int:
        """
        Decodes a DWE binary string back to its integer quantity.
        Returns -1 if the bitstring is an invalid DWE configuration.
        Args:
            bitstring (str): The DWE binary string (e.g., "00", "10", "11").
        Returns:
            int: The decoded integer quantity, or -1 if invalid.
        """
        if len(bitstring)!= self.num_qubits:
            return -1 # Invalid length

        # Check for valid DWE pattern (contiguous 1s followed by contiguous 0s)
        # Example: "00", "10", "11" are valid. "01" is invalid.
        try:
            first_zero_idx = bitstring.find('0')
            if first_zero_idx == -1: # All ones, e.g., "11" for max_q=1, or "111" for max_q=2
                return self.num_qubits
            
            # Check if all subsequent characters are '0'
            if '1' in bitstring[first_zero_idx:]:
                return -1 # Invalid pattern like "101" or "010"
            
            return first_zero_idx # Number of leading ones is the quantity
        except ValueError:
            return -1 # Should not happen for binary strings, but good practice

    def get_valid_dwe_bitstrings(self) -> list[str]:
        """
        Generates all valid DWE binary strings for the defined max_quantity.
        Returns:
            list[str]: A list of valid DWE bitstrings.
        """
        valid_bitstrings = []
        for q in range(self.max_quantity + 1):
            valid_bitstrings.append(self.encode_quantity(q))
        return valid_bitstrings


def create_dwe_cost_hamiltonian(num_qubits: int, A: float, B: float, dwe_penalty_strength: float) -> SparsePauliOp:
    """
    Constructs the DWE-based Cost Hamiltonian (Hc) for the single-asset quantity problem.
    H_C = (2x_0x_1 - 2x_0 - 2x_1) + lambda_DWE * (1-x_0)x_1
    H_C = -8x_0x_1 - 2x_0 + 8x_1 (after simplification with lambda_DWE=10)
    Converted to Pauli Z: H_C = 3Z_0 - 2Z_1 - 2Z_0Z_1 + 1 (from Part 1 numerical example)

    Args:
        num_qubits (int): Number of qubits for DWE (2 in our example).
        A (float): Coefficient for the quadratic term (q^2).
        B (float): Coefficient for the linear term (-q).
        dwe_penalty_strength (float): Strength of the DWE penalty term.
    Returns:
        SparsePauliOp: The constructed Cost Hamiltonian.
    """
    if num_qubits!= 2:
        raise ValueError("This specific Hc formulation is for 2 DWE qubits (q in {0,1,2})")

    pauli_terms = [
        (Pauli('Z') @ Pauli('I'), 3.0),
        (Pauli('I') @ Pauli('Z'), -2.0),
        (Pauli('Z') @ Pauli('Z'), -2.0),
        (Pauli('I') @ Pauli('I'), 1.0)
    ]
    
    hc_pauli_list = [
        ('ZI', 3.0),
        ('IZ', -2.0),
        ('ZZ', -2.0),
        ('II', 1.0)
    ]

    return SparsePauliOp.from_list(hc_pauli_list)

def create_dwe_mixer_hamiltonian(num_qubits: int) -> SparsePauliOp:
    """
    Constructs the DWE-aligned Mixer Hamiltonian (Hm) for the single-asset quantity problem.
    H_M = 1/2 (X_0 + X_0 Z_1 + X_1 - X_1 Z_0) (from Part 1 numerical example)

    Args:
        num_qubits (int): Number of qubits for DWE (2 in our example).
    Returns:
        SparsePauliOp: The constructed Mixer Hamiltonian.
    """
    if num_qubits!= 2:
        raise ValueError("This specific Hm formulation is for 2 DWE qubits.")

    hm_pauli_list = [
        ('XI', 0.5),
        ('XZ', 0.5),
        ('IX', 0.5),
        ('ZX', -0.5)
    ]

    return SparsePauliOp.from_list(hm_pauli_list)

def prepare_aligned_initial_state(mixer_hamiltonian: SparsePauliOp) -> QuantumCircuit:
    """
    Prepares a QuantumCircuit that creates the ground state of the given mixer Hamiltonian.
    For small Hamiltonians, this is done by numerical diagonalization.
    Args:
        mixer_hamiltonian (SparsePauliOp): The mixer Hamiltonian.
    Returns:
        QuantumCircuit: A circuit that prepares the ground state of the mixer.
    """
    # Convert SparsePauliOp to a dense matrix for diagonalization
    mixer_matrix = mixer_hamiltonian.to_matrix()

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(mixer_matrix)

    # The ground state corresponds to the smallest eigenvalue
    ground_state_idx = np.argmin(eigenvalues)
    ground_state_vector = eigenvectors[:, ground_state_idx]

    # Create a QuantumCircuit to prepare this state
    num_qubits = mixer_hamiltonian.num_qubits
    initial_state_qc = QuantumCircuit(num_qubits)
    initial_state_qc.initialize(ground_state_vector, range(num_qubits))

    return initial_state_qc

def run_dwe_qaoa(
    num_qubits: int,
    cost_hamiltonian: SparsePauliOp,
    mixer_hamiltonian: SparsePauliOp,
    initial_state_circuit: QuantumCircuit,
    p_layers: int,
    optimizer_method: str = 'COBYLA',
    max_iterations: int = 100
):
    """
    Runs QAOA with custom DWE-based Hamiltonians and an aligned initial state.

    Args:
        num_qubits (int): Total number of qubits.
        cost_hamiltonian (SparsePauliOp): The DWE-based Cost Hamiltonian.
        mixer_hamiltonian (SparsePauliOp): The DWE-aligned Mixer Hamiltonian.
        initial_state_circuit (QuantumCircuit): The circuit to prepare the aligned initial state.
        p_layers (int): The number of QAOA layers (p).
        optimizer_method (str): Classical optimizer method (e.g., 'COBYLA').
        max_iterations (int): Maximum iterations for the classical optimizer.

    Returns:
        dict: Optimization results including optimal parameters, energy, and final state.
    """
    # Initialize Qiskit Sampler primitive
    # Sampler is the recommended primitive for QAOA in Qiskit
    sampler = Sampler()

    # Initialize classical optimizer
    #optimizer = COBYLA(maxiter=max_iterations)
    if optimizer_method.upper() == 'COBYLA':
        optimizer = COBYLA(maxiter=max_iterations)
    elif optimizer_method.upper() == 'SPSA':
        optimizer = SPSA(maxiter=max_iterations)
    elif optimizer_method.upper() == 'L_BFGS_B':
        optimizer = L_BFGS_B(maxiter=max_iterations)
    else:
        raise ValueError(f"Unsupported optimizer method: {optimizer_method}")

    # Initialize QAOA algorithm with custom components
    # The QAOA class from qiskit_algorithms accepts custom mixer and initial_state
    qaoa_instance = QAOA(
        sampler=sampler,
        optimizer=optimizer,
        reps=p_layers,
        initial_state=initial_state_circuit, # Our DWE-aligned initial state
        mixer=mixer_hamiltonian,             # Our custom DWE mixer
        #cost_operator=cost_hamiltonian       # Our custom DWE cost Hamiltonian
    )

    # Compute the minimum eigenvalue (run QAOA optimization)
    # The compute_minimum_eigenvalue method takes the cost_operator
    result = qaoa_instance.compute_minimum_eigenvalue(cost_hamiltonian)

    optimal_params = result.optimal_parameters
    optimal_energy = result.optimal_value
    #print("\n--- QAOA Optimization Results ---")
    #print(result)

    return {
        "optimal_point": result.optimal_point,
        "optimal_energy": result.optimal_value,
        "time_taken": result.optimizer_time,
        "eigenstate": result.eigenstate,
        "best_measurement": result.best_measurement,
        "evaluations": result.cost_function_evals
    }

def get_qaoa_dwe_results(
    A_coeff: int,
    B_coeff: int,
    max_quantity: int = 2,
    DWE_PENALTY_STRENGTH: int = 10,
    p_layers: int = 1,
    optimizer_method: str = 'COBYLA',
    max_iterations: int = 50
):

    # --- DWE Encoding Setup ---
    dwe_encoder = DWEEncoder(max_quantity)
    num_qubits_dwe = dwe_encoder.num_qubits

    # --- Construct Hamiltonians ---
    # Cost Hamiltonian (Hc)
    Hc_dwe = create_dwe_cost_hamiltonian(num_qubits_dwe, A=A_coeff, B=B_coeff, dwe_penalty_strength=DWE_PENALTY_STRENGTH)
    #print("\n--- Constructed DWE Cost Hamiltonian (Hc) ---")
    #print(Hc_dwe)

    # Mixer Hamiltonian (Hm)
    Hm_dwe = create_dwe_mixer_hamiltonian(num_qubits_dwe)
    #print("\n--- Constructed DWE Mixer Hamiltonian (Hm) ---")
    #print(Hm_dwe)

    # --- Prepare Aligned Initial State ---
    initial_state_dwe_aligned = prepare_aligned_initial_state(Hm_dwe)
    #print("\n--- Prepared DWE-aligned Initial State Circuit ---")
    #print(initial_state_dwe_aligned.draw(output='text', idle_wires=False))

    # --- Run QAOA ---
    p_layers = 1 # Number of QAOA layers
    qaoa_results = run_dwe_qaoa(
        num_qubits=num_qubits_dwe,
        cost_hamiltonian=Hc_dwe,
        mixer_hamiltonian=Hm_dwe,
        initial_state_circuit=initial_state_dwe_aligned,
        p_layers=p_layers,
        optimizer_method=optimizer_method,
        max_iterations=50 # Small number of iterations for quick proof-of-concept
    )

    #return qaoa_results

    # --- Analyze Results ---
    probabilities = qaoa_results["eigenstate"]
    optimal_energy = qaoa_results["optimal_energy"]

    #print(f"\n--- Analysis of QAOA Results (p={p_layers}) ---")
    #print(f"Optimal Energy found by QAOA: {optimal_energy:.4f}")

    # Get probabilities from the optimal statevector
    #probabilities = np.abs(optimal_statevector)**2

    #print("\nProbabilities of each computational basis state:")
    decoded_results = {}
    for i in probabilities.keys():
        bitstring = format(i, f'0{num_qubits_dwe}b')
        prob = probabilities[i]
        decoded_q = dwe_encoder.decode_bitstring(bitstring)
        
        # Calculate classical cost for valid states
        classical_cost = None
        if decoded_q!= -1:
            classical_cost = A_coeff * (decoded_q**2) - B_coeff * decoded_q
        
        #print(f"|{bitstring}> (q={decoded_q if decoded_q!= -1 else 'INVALID'}): Probability = {prob:.4f}, Classical Cost = {classical_cost if classical_cost is not None else 'N/A'}")
        
        if decoded_q!= -1:
            decoded_results[bitstring] = {"quantity": decoded_q, "probability": prob, "cost": classical_cost}

    # Identify the most probable valid solution(s)
    most_probable_valid_bitstring = None
    max_prob = -1
    for bs, data in decoded_results.items():
        if data["probability"] > max_prob:
            max_prob = data["probability"]
            most_probable_valid_bitstring = bs

    if most_probable_valid_bitstring:
        final_quantity = decoded_results[most_probable_valid_bitstring]["quantity"]
        final_cost = decoded_results[most_probable_valid_bitstring]["cost"]
        #print(f"\nMost probable valid solution: Quantity = {final_quantity} (bitstring: {most_probable_valid_bitstring})")
        #print(f"Corresponding classical cost: {final_cost}")
    else:
        print("\nNo valid solution found with significant probability.")

    return { 
        "beta": qaoa_results['optimal_point'][0],
        "gamma": qaoa_results['optimal_point'][1],
        "gs_energy": qaoa_results['optimal_energy'],
        "time": qaoa_results['time_taken'],
        "evaluations": qaoa_results['evaluations'],
        "p": p_layers,
        "N": num_qubits_dwe,
        "solution": qaoa_results['best_measurement']['bitstring'],
        "probability": qaoa_results['best_measurement']['probability'],
        "optimizer": optimizer_method
    }
