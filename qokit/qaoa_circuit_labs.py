###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
# QAOA circuit for some Z objective
from collections.abc import Sequence
from qiskit import QuantumCircuit
from .qaoa_circuit import get_qaoa_circuit_from_terms, get_parameterized_qaoa_circuit_from_terms


def get_qaoa_circuit(N: int, terms: Sequence, gamma: Sequence, beta: Sequence, save_statevector: bool = True) -> QuantumCircuit:
    """Generates a circuit for Hamiltonian of the form \sum_{term \in terms} \prod_{j \in term} Z_j

    Parameters
    ----------
    N : int
        Number of qubits
    terms : list of tuples
        Each tuple corresponds to a term \prod_{j \in term} Z_j and contains indices
        Example: for H = Z_0*Z_1 + Z_2*Z_3 + Z_0*Z_2*Z_4, terms = [(0,1), (2,3), (0,2,4)]
        All indices must be less than N
    beta : list-like
        QAOA parameter beta
    gamma : list-like
        QAOA parameter gamma
    save_statevector : bool, default True
        Add save state instruction to the end of the circuit
    Returns
    -------
    qc : qiskit.QuantumCircuit
        Quantum circuit implementing QAOA
    """
    return get_qaoa_circuit_from_terms(N=N, terms=terms, gammas=gamma, betas=beta, save_statevector=save_statevector)


def get_parameterized_qaoa_circuit(N: int, terms: Sequence, p: int, save_statevector: bool = True, return_parameter_vectors: bool = False) -> QuantumCircuit:
    """Generates a parameterized circuit for Hamiltonian of the form \sum_{term \in terms} \prod_{j \in term} Z_j
    This version is recommended for long circuits

    Example usage:
        qc_param = get_parameterized_qaoa_circuit(N, terms, p)
        def f(theta):
            ...
            qc = qc_param.bind_parameters(np.hstack([beta, gamma]))
            sv = backend.run(qc).result().get_statevector()
            ...
        minimize(f, ...)

    Parameters
    ----------
    N : int
        Number of qubits
    terms : list of tuples
        Each tuple corresponds to a term \prod_{j \in term} Z_j and contains indices
        Example: for H = Z_0*Z_1 + Z_2*Z_3 + Z_0*Z_2*Z_4, terms = [(0,1), (2,3), (0,2,4)]
        All indices must be less than N
    p : int
        Number of QAOA layers (number of parameters will be 2*p)
    save_statevector : bool, default True
        Add save state instruction to the end of the circuit
    return_parameter_vectors : bool, default False
        Return ParameterVector for betas and gammas

    Returns
    -------
    qc : qiskit.QuantumCircuit
        Parameterized quantum circuit implementing QAOA
        Parameters are two ParameterVector sorted alphabetically
        (beta first, then gamma). To bind:
        qc.bind_parameters(np.hstack([angles['beta'], angles['gamma']]))
    """
    return get_parameterized_qaoa_circuit_from_terms(
        N=N,
        terms=terms,
        p=p,
        save_statevector=save_statevector,
        return_parameter_vectors=return_parameter_vectors,
    )
