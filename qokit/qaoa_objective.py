###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import numpy as np
from qiskit.providers.aer import AerSimulator
from functools import reduce
import numba.cuda
import warnings
from typing import Optional

from .qaoa_vectorized import get_qaoa_statevector as get_qaoa_statevector_vectorized
from .fur import QAOAFURXSimulatorC, QAOAFURXSimulatorGPU

from .parameter_utils import from_fourier_basis


def get_qaoa_objective(
    N: Optional[int] = None,
    p: Optional[int] = None,
    precomputed_diagonal_hamiltonian=None,
    precomputed_objectives=None,
    precomputed_optimal_bitstrings=None,
    parameterized_circuit=None,
    parameterization: str = "theta",
    objective: str = "expectation",
    simulator: str = "auto",
):
    """Return QAOA objective to be minimized

    Parameters
    ----------
    N : int
        Number of qubits
    p : int
        Number of QAOA layers (number of parameters will be 2*p)
    parameterization : str
        If parameterization == 'theta', then f takes one parameter (beta and gamma concatenated)
        If parameterization == 'gamma beta', then f takes two parameters (beta and gamma)
        For below Fourier parameters, q=p
        If parameterization == 'freq', then f takes one parameter (fourier parameters u and v concatenated)
        If parameterization == 'u v', then f takes two parameters (fourier parameters u and v)
    objective : str
        If objective == 'expectation', then returns f(theta) = - < theta | C_{LABS} | theta > (minus for minimization)
        If objective == 'overlap', then returns f(theta) = 1 - Overlap |<theta|optimal_bitstring>|^2 (1-overlap for minimization)
        If objective == 'expectation and overlap', then returns a tuple (expectation, overlap)
    simulator : str
        If simulator == 'auto', implementation is chosen automatically
            (either the fastest CPU simulator or a GPU simulator if CUDA is available)
        If simulator == 'qiskit', implementation in qaoa_qiskit is used

    Returns
    -------
    f : callable
        Function returning the negative of expected value of QAOA with parameters theta
    """
    if simulator == "auto":
        if numba.cuda.is_available():
            simulator = "nbcufurx"
            warnings.warn(f"CUDA is available, using {simulator} simulator")
        else:
            simulator = "cfurx"
            warnings.warn(f"CUDA is NOT available, using {simulator} simulator")

    # Prepare function that computes objective from precomputed energies / bitstrings
    if objective == "expectation":
        if precomputed_objectives is None:
            raise ValueError(f"precomputed_objectives are required when using the {objective} objective")

        def compute_objective_from_probabilities(probabilities):
            return precomputed_objectives.dot(probabilities)

    elif objective == "overlap":
        if precomputed_optimal_bitstrings is None:
            raise ValueError(f"precomputed_optimal_bitstrings are required when using the {objective} objective")

        # extract locations of the optimal_bitstrings in 2**N
        bitstring_loc = np.array([reduce(lambda a, b: 2 * a + b, x) for x in precomputed_optimal_bitstrings])

        def compute_objective_from_probabilities(probabilities):
            # compute overlap
            overlap = 0
            for i in range(len(bitstring_loc)):
                overlap += probabilities[bitstring_loc[i]]
            return 1 - overlap

    elif objective == "expectation and overlap":
        if precomputed_objectives is None:
            raise ValueError(f"precomputed_objectives are required when using the {objective} objective")
        if precomputed_optimal_bitstrings is None:
            raise ValueError(f"precomputed_optimal_bitstrings are required when using the {objective} objective")

        # extract locations of the optimal_bitstrings in 2**N
        bitstring_loc = np.array([reduce(lambda a, b: 2 * a + b, x) for x in precomputed_optimal_bitstrings])

        def compute_objective_from_probabilities(probabilities):
            # compute energy
            en = precomputed_objectives.dot(probabilities)
            # compute overlap
            overlap = 0
            for i in range(len(bitstring_loc)):
                overlap += probabilities[bitstring_loc[i]]
            return en, 1 - overlap

    else:
        raise ValueError(f"Unknown objective passed to get_qaoa_labs_objective: {objective}, allowed ['expectation', 'overlap', 'expectation and overlap']")

    if simulator == "qiskit":
        # Prepare parameterized circuit
        backend = AerSimulator(method="statevector")

        def probabilities_from_beta_gamma(beta, gamma):
            qc = parameterized_circuit.bind_parameters(np.hstack([beta, gamma]))
            sv = np.asarray(backend.run(qc).result().get_statevector())
            return np.abs(sv) ** 2

    else:
        if simulator == "vectorized":

            def probabilities_from_beta_gamma(beta, gamma):
                sv = get_qaoa_statevector_vectorized(beta, gamma, N=N, precomputed_energies=precomputed_diagonal_hamiltonian)
                return np.abs(sv) ** 2

        elif simulator == "cfurx":
            sim = QAOAFURXSimulatorC(N, precomputed_diagonal_hamiltonian)

            def probabilities_from_beta_gamma(beta, gamma):
                sv = sim.simulate_qaoa(gamma, np.asarray(beta) * 2)
                return sv.get_norm_squared()

        elif simulator == "nbcufurx":
            sim = QAOAFURXSimulatorGPU(N, precomputed_diagonal_hamiltonian)

            def probabilities_from_beta_gamma(beta, gamma):
                return sim.simulate_qaoa(gamma, np.asarray(beta) * 2, return_probabilities=True)

        else:
            raise ValueError(f"Unknown simulator passed to get_qaoa_labs_objective: {simulator}, allowed ['vectorized', 'qiskit', 'cfurx', 'nbcufurx']")

    if parameterization == "theta":

        def f(theta):
            gamma = theta[:p]
            beta = theta[p:]
            probabilities = probabilities_from_beta_gamma(beta, gamma)
            return compute_objective_from_probabilities(probabilities)

    elif parameterization == "gamma beta":

        def f(gamma, beta):
            probabilities = probabilities_from_beta_gamma(beta, gamma)
            return compute_objective_from_probabilities(probabilities)

    elif parameterization == "freq":

        def f(freq):
            u = freq[:p]
            v = freq[p:]
            beta, gamma = from_fourier_basis(u, v)
            probabilities = probabilities_from_beta_gamma(beta, gamma)
            return compute_objective_from_probabilities(probabilities)

    elif parameterization == "u v":

        def f(u, v):
            beta, gamma = from_fourier_basis(u, v)
            probabilities = probabilities_from_beta_gamma(beta, gamma)
            return compute_objective_from_probabilities(probabilities)

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
