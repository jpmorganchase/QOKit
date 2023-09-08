###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from __future__ import annotations
import numpy as np
import networkx as nx
import warnings

from .utils import precompute_energies
from .maxcut import maxcut_obj, get_adjacency_matrix, get_maxcut_terms

from .qaoa_circuit_maxcut import get_parameterized_qaoa_circuit
from .qaoa_objective import get_qaoa_objective


def get_qaoa_maxcut_objective(
    N: int,
    p: int,
    G: nx.Graph | None = None,
    precomputed_cuts: np.ndarray | None = None,
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
    G : nx.Graph
        graph on which MaxCut will be solved
    precomputed_cuts : np.array
        precomputed cuts to compute the QAOA expectation
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

    if precomputed_cuts is not None and G is not None:
        warnings.warn("If precomputed_cuts is passed, G is ignored")

    if precomputed_cuts is None:
        assert G is not None, "G must be passed if precomputed_cuts is None"
        terms = get_maxcut_terms(G)

    if simulator == "qiskit":
        assert G is not None, "G must be passed if simulator == 'qiskit'"
        precomputed_cuts = precompute_energies(maxcut_obj, N, w=get_adjacency_matrix(G))
        parameterized_circuit = get_parameterized_qaoa_circuit(G, p)
    else:
        parameterized_circuit = None

    return get_qaoa_objective(
        N=N,
        p=p,
        precomputed_diagonal_hamiltonian=precomputed_cuts,
        precomputed_objectives=precomputed_cuts,
        terms=terms,
        precomputed_optimal_bitstrings=precomputed_optimal_bitstrings,
        parameterized_circuit=parameterized_circuit,
        parameterization=parameterization,
        objective=objective,
        simulator=simulator,
    )
