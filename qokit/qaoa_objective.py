###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from __future__ import annotations
from calendar import c
import numpy as np
from itertools import combinations
from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import Aer
from functools import reduce
import numba.cuda

from .fur import choose_simulator, choose_simulator_xyring, QAOAFastSimulatorBase
import typing

import qokit.parameter_utils
from qokit.parameter_utils import QAOAParameterization
from .qaoa_circuit_portfolio import measure_circuit
from .utils import reverse_array_index_bit_order

from .fur.diagonal_precomputation import precompute_vectorized_cpu_parallel


def generate_k_hot_superposition(n, k):
    """
    Create a uniform superposition statevector
    over all bitstrings of Hamming weight k.
    """
    indices = list(combinations(range(n), k))
    dim = 2 ** n
    state = np.zeros(dim, dtype=complex)
    
    for idx in indices:
        bitstring = ['0'] * n
        for i in idx:
            bitstring[i] = '1'
        bin_index = int(''.join(bitstring), 2)
        state[bin_index] = 1.0 / np.sqrt(len(indices))
    
    return state


def build_swap_mixer_layer(n_qubits, beta_value):
    """
    Build a single SWAP-based mixer layer circuit:
    exp(-i beta * (X_i X_j + Y_i Y_j))
    for all unique pairs (i,j).
    """
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        for j in range(i+1, n_qubits):
            qc.rxx(2 * beta_value, i, j)
            qc.ryy(2 * beta_value, i, j)
    return qc


def _get_qiskit_objective(
    parameterized_circuit,
    precomputed_objectives=None,
    precomputed_optimal_bitstrings=None,
    objective: str = "expectation",
    terms=None,
    parameterization: str | QAOAParameterization = "theta",
    mixer: str = "x",
    optimization_type="min",
):
    N = parameterized_circuit.num_qubits
    if objective == "expectation":
        if precomputed_objectives is None:
            if terms is None:
                raise ValueError(f"precomputed_objectives or terms are required when using the {objective} objective")
            else:
                precomputed_objectives = precompute_vectorized_cpu_parallel(terms, 0.0, N)

        def compute_objective_from_probabilities(probabilities):  # type: ignore
            if optimization_type == "max":
                return -1 * precomputed_objectives.dot(probabilities)
            return precomputed_objectives.dot(probabilities)

    elif objective == "overlap":
        if precomputed_optimal_bitstrings is None:
            if precomputed_objectives is None:
                if terms is None:
                    raise ValueError(f"precomputed_objectives or terms are required when using the {objective} objective")
                else:
                    precomputed_objectives = precompute_vectorized_cpu_parallel(terms, 0.0, N)
            if optimization_type == "max":
                precomputed_objectives = -1 * np.asarray(precomputed_objectives)
            minval = precomputed_objectives.min()
            bitstring_loc = (precomputed_objectives == minval).nonzero()
            assert len(bitstring_loc) == 1
            bitstring_loc = bitstring_loc[0]
        else:
            # extract locations of the optimal_bitstrings in 2**N
            bitstring_loc = np.array([reduce(lambda a, b: 2 * a + b, x) for x in precomputed_optimal_bitstrings])

        def compute_objective_from_probabilities(probabilities):  # type: ignore
            # compute overlap
            overlap = 0
            for i in range(len(bitstring_loc)):
                overlap += probabilities[bitstring_loc[i]]
            return 1 - overlap

    else:
        raise ValueError(f"Unknown objective passed to get_qaoa_objective: {objective}, allowed ['expectation', 'overlap']")

    if mixer == "x":
        backend = Aer.get_backend("aer_simulator_statevector")

        def g(gamma, beta):
            qc = parameterized_circuit.assign_parameters(list(np.hstack([beta, gamma])))
            sv = np.asarray(backend.run(qc).result().get_statevector())
            probs = np.abs(sv) ** 2
            return compute_objective_from_probabilities(probs)

    elif mixer == "xy":
        backend = Aer.get_backend("statevector_simulator")

        def g(gamma, beta):
            qc = parameterized_circuit.assign_parameters(np.hstack([np.asarray(beta) / 2, np.asarray(gamma) / 2]))
            circ = transpile(qc, backend)
            sv = reverse_array_index_bit_order(Statevector(circ))
            probs = np.abs(sv) ** 2
            return compute_objective_from_probabilities(probs)

    else:
        raise ValueError(f"Unknown mixer type passed to get_qaoa_objective: {mixer}, allowed ['x', 'xy']")

    return g


def get_qaoa_objective(
    N: int,
    precomputed_diagonal_hamiltonian=None,
    precomputed_costs=None,
    terms=None,
    precomputed_optimal_bitstrings=None,
    parameterization: str | QAOAParameterization = "theta",
    objective: str = "expectation",
    parameterized_circuit=None,
    simulator: str = "auto",
    mixer: str = "x",
    initial_state: np.ndarray | None = None,
    n_trotters: int = 1,
    optimization_type="min",
    simulator_kw: dict | None = None,
    k_hot: int | None = None,
) -> typing.Callable:
    """Return QAOA objective to be minimized

    Parameters
    ----------
    N : int
        Number of qubits
    p : int
        Number of QAOA layers (number of parameters will be 2*p)
    parameterization : str
        If parameterization == 'theta', then f takes one parameter (gamma and beta concatenated)
        If parameterization == 'gamma beta', then f takes two parameters (gamma and beta)
        For below Fourier parameters, q=p
        If parameterization == 'freq', then f takes one parameter (fourier parameters u and v concatenated)
        If parameterization == 'u v', then f takes two parameters (fourier parameters u and v)
    objective : str
        If objective == 'expectation', then returns f(theta) = - < theta | C_{LABS} | theta > (minus for minimization)
        If objective == 'overlap', then returns f(theta) = 1 - Overlap |<theta|optimal_bitstring>|^2 (1-overlap for minimization)
    simulator : str
        If simulator == 'auto', implementation is chosen automatically
            (either the fastest CPU simulator or a GPU simulator if CUDA is available)
        If simulator == 'qiskit', implementation in qaoa_qiskit is used
    mixer : str
        If mixer == 'x', then uses the default Pauli X as the mixer
        If mixer == 'xy', then uses the ring-XY as the mixer
        If mixer == 'swap' than uses SWAP-based Hamming-weight preserving mixer
    initial_state : np.ndarray
        The initial state for QAOA, default is the uniform superposition state (corresponding to the X mixer)
    n_trotters : int
        Number of Trotter steps in each mixer layer for the xy mixer
    k_hot : int, optional
        If set, builds uniform superposition of all bitstrings of Hamming weight k as initial state

    Returns
    -------
    f : callable
        Function returning the negative of expected value of QAOA with parameters theta
    """

    # -- Qiskit edge case
    if simulator == "qiskit":
        if precomputed_costs is None:
            precomputed_costs = precomputed_diagonal_hamiltonian
        g = _get_qiskit_objective(
            parameterized_circuit,
            precomputed_costs,
            precomputed_optimal_bitstrings,
            objective,
            terms,
            parameterization,
            mixer,
            optimization_type=optimization_type,
        )

        def fq(*args):
            gamma, beta = qokit.parameter_utils.convert_to_gamma_beta(*args, parameterization=parameterization)
            return g(gamma, beta)

        return fq
    # --

    if mixer == "x":
        simulator_cls = choose_simulator(name=simulator)
    elif mixer in ["xy", "swap"]:
        simulator_cls = choose_simulator_xyring(name=simulator)
    else:
        raise ValueError(f"Unknown mixer type passed to get_qaoa_objective: {mixer}, allowed ['x', 'xy', 'swap']")

    sim = simulator_cls(N, terms=terms, costs=precomputed_diagonal_hamiltonian, **(simulator_kw or {}))
    if precomputed_costs is None:
        precomputed_costs = sim.get_cost_diagonal()

    bitstring_loc = None
    if precomputed_optimal_bitstrings is not None and objective != "expectation":
        bitstring_loc = np.array([reduce(lambda a, b: 2 * a + b, x) for x in precomputed_optimal_bitstrings])

    # Prepare initial state
    if k_hot is not None:
        psi0 = generate_k_hot_superposition(N, k_hot)
    elif initial_state is None:
        psi0 = np.zeros(2 ** N, dtype=complex)
        psi0[0] = 1
    else:
        psi0 = initial_state

    # -- Final function
    def f(*args, **kw):
        gamma, beta = qokit.parameter_utils.convert_to_gamma_beta(*args, parameterization=parameterization)
        # Build custom mixers for SWAP
        if mixer == "swap":
            mixers = []
            for b in beta:
                mixers.append(build_swap_mixer_layer(N, b))
        else:
            mixers = None
        result = sim.simulate_qaoa(gamma,beta,psi0,mixer_circuits=mixers,n_trotters=n_trotters, **kw)
        if objective == "expectation":
            return sim.get_expectation(result, costs=precomputed_costs, preserve_state=False, optimization_type=optimization_type)
        elif objective == "overlap":
            overlap = sim.get_overlap(result, costs=precomputed_costs, indices=bitstring_loc, preserve_state=False, optimization_type=optimization_type)
            return 1 - overlap

    return f


def get_qaoa_labs_overlap(N: int, p: int, **kwargs):
    """To be deprecated. Use get_qaoa_labs_objective going forward.
    Please consult the docstring for get_qaoa_labs_objective for kwargs
    Return:
    1 - Overlap |<theta|optimal_bitstring>|^2

    Parameters
    ----------
    N : int
        Number of qubits
    p : int
        Number of QAOA layers (number of parameters will be 2*p)

    Returns
    -------
    f : callable
        Retrun the 1 - overlap of the state with the optimal bitstrings (we compute the sum of the probability of the output state to be in any one of the optimal bitstring states).
    """
    return get_qaoa_labs_objective(N, p, objective="overlap", **kwargs)


def get_qaoa_labs_overlap_fourier(N: int, p: int, parameterization: str = "freq", **kwargs):
    """To be deprecated. Use get_qaoa_labs_objective going forward.
    Please consult the docstring for get_qaoa_labs_objective for kwargs

    Parameters
    ----------
    N : int
        Number of qubits
    p : int
        Number of QAOA layers (number of parameters will be 2*p)
    parameterization : str
        If parameterization == 'freq', then f takes one parameter
        If parameterization == 'u v', then f takes two parameters (u and v)
    Returns
    -------
    f : callable
        Retrun the 1 - overlap of the state with the optimal bitstrings (we compute the sum of the probability of the output state to be in any one of the optimal bitstring states).
    """

    return get_qaoa_labs_objective(N, p, objective="overlap", parameterization=parameterization, **kwargs)
