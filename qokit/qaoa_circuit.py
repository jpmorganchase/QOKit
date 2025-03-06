###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
# QAOA circuit for S_k

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from typing import Sequence


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
    if term_weight == 4:
        # in labs, four-body terms appear two times more than two-body
        # there is also a global scaling factor of 2 for all terms (four and two), which is ignored here
        assert all(term[i] < term[i + 1] for i in range(len(term) - 1))
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


def append_x_term(qc: QuantumCircuit, q1: int, beta: float) -> None:
    qc.rx(2 * beta, q1)


def append_cost_operator_circuit(qc: QuantumCircuit, terms: Sequence, gamma: float) -> None:
    for term in terms:
        if len(term) == 2 and isinstance(term[1], tuple):
            coeff, (i, j) = term
            append_z_prod_term(qc, (i, j), gamma * coeff / 2)
        elif any([isinstance(i, tuple) for i in term]):
            raise ValueError(f"Invalid term received: {term}")
        else:
            append_z_prod_term(qc, term, gamma)


def append_mixer_operator_circuit(qc: QuantumCircuit, beta: float) -> None:
    for n in qc.qubits:
        append_x_term(qc, n, beta)


def get_qaoa_circuit_from_terms(
    N: int, terms: Sequence, gammas: Sequence, betas: Sequence, save_statevector: bool = True, qr: QuantumRegister = None, cr: ClassicalRegister = None
):
    """Generates a circuit for weighted MaxCut, or SK problem.
    Parameters
    ----------
    terms : list-like
        Each element corresponds to terms in the Hamiltonian
        and pair of spins or graph nodes.
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
        append_cost_operator_circuit(qc, terms, gammas[i])
        append_mixer_operator_circuit(qc, betas[i])
    if save_statevector:
        qc.save_statevector()
    return qc


def get_parameterized_qaoa_circuit_from_terms(
    N: int,
    terms: Sequence,
    p: int,
    save_statevector: bool = True,
    qr: QuantumRegister = None,
    cr: ClassicalRegister = None,
    return_parameter_vectors: bool = False,
):
    """Generates a parameterized circuit for weighted MaxCut on graph G.
    This version is recommended for long circuits

    Parameters
    ----------
    terms : list-like
        Each element corresponds to terms in the Hamiltonian
        and pair of spins or graph nodes.
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
        append_cost_operator_circuit(qc, terms, gammas[i])
        append_mixer_operator_circuit(qc, betas[i])
    if save_statevector:
        qc.save_statevector()
    if return_parameter_vectors:
        return qc, betas, gammas
    else:
        return qc
