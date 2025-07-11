###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import qiskit
import numpy as np
from .portfolio_optimization import yield_all_indices_cosntrained, get_configuration_cost
from qiskit import QuantumCircuit, transpile, QuantumRegister
from qiskit_aer import Aer
from qiskit.circuit import ParameterVector
from .utils import reverse_array_index_bit_order, state_to_ampl_counts

from qiskit.quantum_info import Statevector
from qiskit.circuit.library import RXXGate, RYYGate # Import if not already present

def generate_dicke_state_fast(N, K):
    """
    Generate the dicke state with yield function
    Returns a contiguous np.complex128 array for performance.
    """
    index = yield_all_indices_cosntrained(N, K)
    s = np.zeros(2**N, dtype=np.complex128)
    for i in index:
        s[i] = 1.0
    s = 1 / np.sqrt(np.sum(s)) * s
    return np.ascontiguousarray(s)


# def get_cost_circuit(po_problem, qc, gamma):
#     """
#     Construct the problem Hamiltonian layer for QAOA circuit
#     H = 0.5*q\sum_{i=1}^{n-1} \sum_{j=i+1}^n \sigma_{ij}Z_i Z_j + 0.5 \sum_i (-q\sum_{j=1}^n{\sigma_ij} + \mu_i) Z_i + Constant
#     """
#     q = po_problem["q"]
#     means = po_problem["means"]
#     cov = po_problem["cov"]
#     N = po_problem["N"]
#     for i in range(N):
#         qc.rz((means[i] - q * np.sum(cov[i, :])) * gamma, i)  # there is a 0.5 inside rz and rzz
#     for i in range(N - 1):
#         for j in range(i + 1, N):
#             qc.rzz(q * cov[i, j] * gamma, i, j)
#     return qc
def get_cost_circuit(qr, J_coeffs, h_coeffs, gamma, T=1): # <<-- IMPORTANT: Change signature to take J_coeffs, h_coeffs
    qc_cost = QuantumCircuit(qr)
    # Apply RZ gates for linear terms (h_i Z_i)
    for i in range(qr.size):
        if i in h_coeffs and h_coeffs[i] != 0:
            qc_cost.rz(2 * h_coeffs[i] * gamma * T, qr[i]) # Rz expects 2 * angle * time

    # Apply RZZ gates for quadratic terms (J_ij Z_i Z_j)
    for (i, j), val in J_coeffs.items():
        if val != 0:
            # Ensure i < j for consistency if J_coeffs only has (i,j) where i<j
            # If J_coeffs might have (j,i) as well, normalize it or iterate appropriately
            qc_cost.rzz(2 * val * gamma * T, qr[i], qr[j]) # Rzz expects 2 * angle * time
    return qc_cost

# The get_parameterized_qaoa_circuit should then call it like:
# qc_qaoa.compose(get_cost_circuit(qr, po_problem["J"], po_problem["h"], gamma[layer], T=T), inplace=True)


def get_dicke_init(N, K):
    """
    Generate dicke state in gates
    """
    from qokit.dicke_state_utils import dicke_simple

    # can be other dicke state implementaitons
    return dicke_simple(N, K)


def get_mixer_Txy(qubits, beta, T=1, mixer_topology='complete'):
    """
    Generates the Trotterized XY Mixer Hamiltonian circuit (e^{-i beta H_M T}).

    Args:
        qubits (list): List of Qiskit QuantumRegister qubits.
        beta (float): Parameter for the mixer Hamiltonian.
        T (float): Trotterization step (usually 1 for full Trotter step).
        mixer_topology (str): Defines the connectivity for XY interactions.
                              'complete': All-to-all connectivity (default, for small N).
                              'linear': Linear chain connectivity (i, i+1).
                              'ring': Ring connectivity (i, i+1 and N-1, 0).
    Returns:
        QuantumCircuit: Circuit for the mixer operation.
    """
    qc_mixer = QuantumCircuit(qubits)
    N = len(qubits)

    if mixer_topology == 'complete':
        # This is your current all-to-all implementation
        for i in range(N):
            for j in range(i + 1, N):
                qc_mixer.rxx(2 * beta * T, qubits[i], qubits[j])
                qc_mixer.ryy(2 * beta * T, qubits[i], qubits[j])
    elif mixer_topology == 'linear':
        # Linear chain: (0,1), (1,2), ..., (N-2, N-1)
        for i in range(N - 1):
            qc_mixer.rxx(2 * beta * T, qubits[i], qubits[i+1])
            qc_mixer.ryy(2 * beta * T, qubits[i], qubits[i+1])
    elif mixer_topology == 'ring':
        # Ring: (0,1), ..., (N-2, N-1), (N-1, 0)
        for i in range(N):
            j = (i + 1) % N
            qc_mixer.rxx(2 * beta * T, qubits[i], qubits[j])
            qc_mixer.ryy(2 * beta * T, qubits[i], qubits[j])
    else:
        raise ValueError(f"Unknown mixer_topology: {mixer_topology}. Choose from 'complete', 'linear', 'ring'.")

    return qc_mixer


def get_mixer_RX(qc, beta):
    """A layer of RX gates"""
    N = len(qc._qubits)
    for i in range(N):
        qc.rx(2 * beta, i)
    return qc


def get_qaoa_circuit(
    po_problem,
    gammas,
    betas,
    depth,
    ini="dicke",
    mixer="trotter_ring",
    T=1,
    ini_state=None,
    save_state=True,
    minus=False,
):
    """
    Put all ingredients together to build up a qaoa circuit
    Minus is for define mixer with a minus sign, for checking phase diagram

    po_problem is generated by qokit.portfolio_optimization.get_problem
    """
    N = po_problem["N"]
    K = po_problem["K"]
    if ini_state is not None:
        q = QuantumRegister(N)
        circuit = QuantumCircuit(q)
        circuit.initialize(ini_state, [q[i] for i in range(N)])
    else:
        if ini.lower() == "dicke":
            circuit = get_dicke_init(N, K)
        elif ini.lower() == "uniform":
            circuit = get_uniform_init(N)
        else:
            raise ValueError("Undefined initial circuit")
    for i in range(depth):
        circuit = get_cost_circuit(po_problem, circuit, gammas[i])
        if mixer.lower() == "trotter_ring":
            circuit = get_mixer_Txy(circuit, betas[i], minus=minus, T=T)  # minus should be false
        elif mixer.lower() == "rx":
            circuit = get_mixer_RX(circuit, betas[i])
        else:
            raise ValueError("Undefined mixer circuit")
    if save_state is False:
        circuit.measure_all()
    return circuit


def get_parameterized_qaoa_circuit(po_problem, depth=1, ini_type='uniform', mixer_type='trotter_ring', T=1, simulator='qiskit', mixer_topology='complete', gamma=None, beta=None): # <--- ADD gamma and beta
    """
    Returns the parameterized QAOA circuit for portfolio optimization.
    # ... (rest of docstring) ...
    """
    N = po_problem["N"]
    qr = QuantumRegister(N, 'q') # <--- ADD THIS LINE! Define the quantum register
    qc_qaoa = QuantumCircuit(qr) # <--- ADD/ENSURE THIS LINE! Initialize the quantum circuit

    if ini_type == 'uniform':
        # Apply Hadamard gates to all qubits for uniform superposition
        for i in range(N):
            qc_qaoa.h(qr[i])
    elif ini_type == 'dicke':
        # Use get_dicke_init to prepare the Dicke state
        # K is needed for Dicke state, get it from po_problem
        K = po_problem["K"] # <--- Ensure K is retrieved here if not already
        qc_initial = get_dicke_init(N, K=K) # <<-- PASS N (the integer) instead of qr
        qc_qaoa.compose(qc_initial, inplace=True)
    else:
        raise ValueError(f"Unknown initial state type: {ini_type}")
    
    
    # QAOA layers
    for layer in range(depth):
        # Cost Hamiltonian (H_C)
        # This part remains the same, as the DWE penalty is baked into po_problem['J'] and po_problem['h']
        qc_qaoa.compose(get_cost_circuit(qr, po_problem["J"], po_problem["h"], gamma[layer], T=T), inplace=True)

        # Mixer Hamiltonian (H_M)
        if mixer_type == 'trotter_ring':
            # Pass the new mixer_topology argument here
            qc_qaoa.compose(get_mixer_Txy(qr, beta[layer], T=T, mixer_topology=mixer_topology), inplace=True) # Ensure beta[layer] is used
        elif mixer_type == 'x_mixer':
            qc_qaoa.compose(get_mixer_x(qr, beta[layer], T=T), inplace=True) # Ensure beta[layer] is used
        else:
            raise ValueError(f"Unknown mixer type: {mixer_type}")

    return qc_qaoa
    

def get_energy_expectation(po_problem, samples):
    """Compute energy expectation from measurement samples"""
    expectation_value = 0
    N_total = 0
    for config, count in samples.items():
        expectation_value += count * get_configuration_cost(po_problem, config)
        N_total += count
    expectation_value = expectation_value / N_total

    return expectation_value


def get_energy_expectation_sv(po_problem, samples):
    """Compute energy expectation from full state vector"""
    expectation_value = 0
    # convert state vector to dictionary
    samples = state_to_ampl_counts(samples)
    for config, wf in samples.items():
        expectation_value += (np.abs(wf) ** 2) * get_configuration_cost(po_problem, config)

    return expectation_value


def invert_counts(counts):
    """convert qubit order for measurement samples"""
    return {k[::-1]: v for k, v in counts.items()}


def measure_circuit(circuit, n_trials=1024, save_state=True):
    """Get the output from circuit, either measured samples or full state vector"""
    if save_state is False:
        backend = Aer.get_backend("qasm_simulator")
        job = transpile(circuit, backend, shots=n_trials)
        result = job.result()
        bitstrings = invert_counts(result.get_counts())
        return bitstrings
    else:
        backend = Aer.get_backend("statevector_simulator")
        circ = transpile(circuit, backend)
        state = Statevector(circ)
        return reverse_array_index_bit_order(state)


def circuit_measurement_function(
    po_problem,
    p,
    ini="dicke",
    mixer="trotter_ring",
    T=None,
    ini_state=None,
    n_trials=1024,
    save_state=True,
    minus=False,
):
    """Helper function to define the objective function to optimize"""

    def f(x):
        gammas = x[0:p]
        betas = x[p:]
        circuit = get_qaoa_circuit(
            po_problem,
            ini=ini,
            mixer=mixer,
            T=T,
            ini_state=ini_state,
            gammas=gammas,
            betas=betas,
            depth=p,
            save_state=save_state,
            minus=minus,
        )
        samples = measure_circuit(circuit, n_trials=n_trials, save_state=save_state)
        if save_state is False:
            energy_expectation_value = get_energy_expectation(po_problem, samples)
        else:
            energy_expectation_value = get_energy_expectation_sv(po_problem, samples)
        return energy_expectation_value

    return f
