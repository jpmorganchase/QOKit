###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import numpy as np
import warnings

from .utils import precompute_energies, reverse_array_index_bit_order
from .sk import sk_obj, get_sk_terms

from .qaoa_circuit_sk import get_parameterized_qaoa_circuit
from .qaoa_objective import get_qaoa_objective


def get_qaoa_sk_objective(
    N: int,
    p: int,
    J: np.ndarray,
    precomputed_energies: np.ndarray | None = None,
    parameterization: str = "theta",
    objective: str = "expectation",
    precomputed_optimal_bitstrings: np.ndarray | None = None,
    simulator: str = "auto",
):
    """Return QAOA objective to be minimized

    Parameters
    ----------
    N : int
        Number of qubits
    p : int
        Number of QAOA layers (number of parameters will be 2*p)
    J : numpy.ndarray
        Couplinf matrix for the SK model.
    precomputed_energies : np.array
        precomputed cuts to compute the QAOA expectation, for maximization problem
        send the precomputed cuts/energies as negative
    parameterization : str
        If parameterization == 'theta', then f takes one parameter (gamma and beta concatenated)
        If parameterization == 'gamma beta', then f takes two parameters (gamma and beta)
        For below Fourier parameters, q=p
        If parameterization == 'freq', then f takes one parameter (fourier parameters u and v concatenated)
        If parameterization == 'u v', then f takes two parameters (fourier parameters u and v)
    precomputed_optimal_bitstrings : np.ndarray
        precomputed optimal bit strings to compute the QAOA overlap
    simulator : str
        If simulator == 'auto', implementation is chosen automatically
            (either the fastest CPU simulator or a GPU simulator if CUDA is available)

    Returns
    -------
    f : callable
        Function returning the negative of expected value of QAOA with parameters theta
    """
    terms = None
    optimization_type = "min"

    if precomputed_energies is not None and J is not None:
        warnings.warn("If precomputed_energies is passed, J is ignored")

    if precomputed_energies is None:
        assert J is not None, "J must be passed if precomputed_energies is None"
        terms = get_sk_terms(J)

    if simulator == "qiskit":
        assert J is not None, "J must be passed if simulator == 'qiskit'"
        precomputed_energies = precompute_energies(sk_obj, N, J)
        parameterized_circuit = get_parameterized_qaoa_circuit(J, p)
    else:
        parameterized_circuit = None

    return get_qaoa_objective(
        N=N,
        precomputed_diagonal_hamiltonian=precomputed_energies,
        terms=terms,
        precomputed_optimal_bitstrings=precomputed_optimal_bitstrings,
        parameterized_circuit=parameterized_circuit,
        parameterization=parameterization,
        objective=objective,
        simulator=simulator,
        optimization_type=optimization_type,
    )
