###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
# QAOA circuit for S_k

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from typing import Sequence


def append_zz_term(qc, q1, q2, gamma):
    qc.rzz(-gamma / 2, q1, q2)


def append_z_prod_term(qc: QuantumCircuit, term: Sequence, gamma: float) -> None:
    """Appends a  multi-body Pauli-Z interaction acting on qubits whose indices
    correspond to those in 'term'.

    Parameters:
        qc: QuantumCircuit
        term: iterable
            ordered iterable containing qubit indices to apply Pauli-Z interaction to
        gamma: float
            evolution time for interaction

    """
    # term_weight, term = term
    term_weight = len(term)
    assert all(term[i] < term[i + 1] for i in range(len(term) - 1))
    if term_weight == 4:
        # in labs, four-body terms appear two times more than two-body
        # there is also a global scaling factor of 2 for all terms (four and two), which is ignored here
        _gamma = 2 * gamma
        qc.cx(term[0], term[1])
        qc.cx(term[3], term[2])
        qc.rzz(2 * _gamma, term[1], term[2])
        qc.cx(term[3], term[2])
        qc.cx(term[0], term[1])
    elif term_weight == 2:
        qc.rzz(2 * gamma, term[0], term[1])
    else:
        # fallback to general case
        target = term[-1]
        for control in term[:-1]:
            qc.cx(control, target)
        qc.rz(2 * gamma, target)
        for control in term[:-1]:
            qc.cx(control, target)


def append_cost_operator_circuit(qc, J, gamma):
    N = J.shape[0]
    for i in range(N):
        for j in range(i + 1, N):
            append_zz_term(qc, i, j, gamma * J[i][j])


def append_labs_cost_operator_circuit(qc: QuantumCircuit, terms: Sequence, gamma: float) -> None:
    for term in terms:
        append_z_prod_term(qc, term, gamma)


def append_x_term(qc, q1, beta):
    qc.rx(2 * beta, q1)


def append_labs_x_term(qc: QuantumCircuit, q1, beta: float) -> None:
    qc.h(q1)
    qc.rz(2 * beta, q1)
    qc.h(q1)


def append_mixer_operator_circuit(qc, N, beta):
    for n in range(N):
        append_x_term(qc, n, beta)


def append_labs_mixer_operator_circuit(qc: QuantumCircuit, beta: float) -> None:
    for n in qc.qubits:
        append_labs_x_term(qc, n, beta)


def _get_qaoa_circuit(
    J: np.ndarray, gammas: Sequence, betas: Sequence, save_statevector: bool = True, qr: QuantumRegister = None, cr: ClassicalRegister = None
):
    """Generates a circuit for weighted MaxCut on graph G, or for coupling matrix J for SK problem.
    Parameters
    ----------
    J : numpy.ndarray
        Matrix representing couplings in the SK model.
        Or, Adjacency matrix for the Graph for Maxcut problem.
    beta : list-like
        QAOA parameter beta
    gamma : list-like
        QAOA parameter gamma
    save_statevector : bool, default True
        Add save state instruction to the end of the circuit
    qr : qiskit.QuantumRegister, default None
        Registers to use for the circuit.
        Useful when one has to compose circuits in a complicated way
        By default, G.number_of_nodes() registers are used
    cr : qiskit.ClassicalRegister, default None
        Classical registers, useful if measuring
        By default, no classical registers are added
    Returns
    -------
    qc : qiskit.QuantumCircuit
        Quantum circuit implementing QAOA
    """
    assert len(betas) == len(gammas)
    p = len(betas)  # infering number of QAOA steps from the parameters passed
    N = J.shape[0]
    if qr is not None:
        assert qr.size >= N
    else:
        qr = QuantumRegister(N)

    if cr is not None:
        qc = QuantumCircuit(qr, cr)
    else:
        qc = QuantumCircuit(qr)

    # first, apply a layer of Hadamards
    qc.h(range(N))
    # second, apply p alternating operators
    for i in range(p):
        append_cost_operator_circuit(qc, J, gammas[i])
        append_mixer_operator_circuit(qc, N, betas[i])
    if save_statevector:
        qc.save_statevector()
    return qc


def _get_parameterized_qaoa_circuit(
    J: np.ndarray, p: int, save_statevector: bool = True, qr: QuantumRegister = None, cr: ClassicalRegister = None, return_parameter_vectors: bool = False
):
    """Generates a parameterized circuit for weighted MaxCut on graph G.
    This version is recommended for long circuits

    Parameters
    ----------
    J : numpy.ndarray
        Matrix representing couplings in the SK model.
        Or, Adjacency matrix for the Graph for Maxcut problem.
    p : int
        Number of QAOA layers (number of parameters will be 2*p)
    save_statevector : bool, default True
        Add save state instruction to the end of the circuit
    qr : qiskit.QuantumRegister, default None
        Registers to use for the circuit.
        Useful when one has to compose circuits in a complicated way
        By default, G.number_of_nodes() registers are used
    cr : qiskit.ClassicalRegister, default None
        Classical registers, useful if measuring
        By default, no classical registers are added
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
    N = J.shape[0]
    if qr is not None:
        assert qr.size >= N
    else:
        qr = QuantumRegister(N)

    if cr is not None:
        qc = QuantumCircuit(qr, cr)
    else:
        qc = QuantumCircuit(qr)

    betas = ParameterVector("beta", p)
    gammas = ParameterVector("gamma", p)

    # first, apply a layer of Hadamards
    qc.h(range(N))
    # second, apply p alternating operators
    for i in range(p):
        append_cost_operator_circuit(qc, J, gammas[i])
        append_mixer_operator_circuit(qc, N, betas[i])
    if save_statevector:
        qc.save_statevector()
    if return_parameter_vectors:
        return qc, betas, gammas
    else:
        return qc


def _get_qaoa_circuit_labs(N: int, terms: Sequence, gamma: Sequence, beta: Sequence, save_statevector: bool = True) -> QuantumCircuit:
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
    assert len(beta) == len(gamma)
    p = len(beta)  # infering number of QAOA steps from the parameters passed

    qc = QuantumCircuit(N)

    # first, apply a layer of Hadamards
    qc.h(range(N))
    # second, apply p alternating operators
    for i in range(p):
        append_labs_cost_operator_circuit(qc, terms, gamma[i])
        append_labs_mixer_operator_circuit(qc, beta[i])
    if save_statevector:
        qc.save_statevector()  # type: ignore
    return qc


def _get_parameterized_qaoa_circuit_labs(
    N: int, terms: Sequence, p: int, save_statevector: bool = True, return_parameter_vectors: bool = False
) -> QuantumCircuit:
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
    qc = QuantumCircuit(N)

    betas = ParameterVector("beta", p)
    gammas = ParameterVector("gamma", p)

    # first, apply a layer of Hadamards
    qc.h(range(N))
    # second, apply p alternating operators
    for i in range(p):
        append_labs_cost_operator_circuit(qc, terms, gammas[i])  # type: ignore
        append_labs_mixer_operator_circuit(qc, betas[i])  # type: ignore
    if save_statevector:
        qc.save_statevector()  # type: ignore
    if return_parameter_vectors:
        return qc, betas, gammas
    else:
        return qc
