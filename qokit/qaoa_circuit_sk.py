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


def append_sk_cost_operator_circuit(qc, J, gamma):
    N = J.shape[0]
    for i in range(N):
        for j in range(i + 1, N):
            append_zz_term(qc, i, j, gamma * J[i][j])


def append_x_term(qc, q1, beta):
    qc.rx(2 * beta, q1)


def append_mixer_operator_circuit(qc, J, beta):
    for n in range(J.shape[0]):
        append_x_term(qc, n, beta)


def get_qaoa_circuit(J: np.ndarray, gammas: Sequence, betas: Sequence, save_statevector: bool = True, qr: QuantumRegister = None, cr: ClassicalRegister = None):
    """Generates a circuit for SK model for given coupling matrix J.
    Parameters
    ----------
    J : numpy.ndarray
        Matrix representing couplings in the SK model.
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
        append_sk_cost_operator_circuit(qc, J, gammas[i])
        append_mixer_operator_circuit(qc, J, betas[i])
    if save_statevector:
        qc.save_statevector()
    return qc


def get_parameterized_qaoa_circuit(
    J: np.ndarray, p: int, save_statevector: bool = True, qr: QuantumRegister = None, cr: ClassicalRegister = None, return_parameter_vectors: bool = False
):
    """Generates a parameterized circuit for SK model for given coupling matrix J.
    This version is recommended for long circuits

    Parameters
    ----------
    J : numpy.ndarray
        Coupling matrix for the SK model.
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
        append_sk_cost_operator_circuit(qc, J, gammas[i])
        append_mixer_operator_circuit(qc, J, betas[i])
    if save_statevector:
        qc.save_statevector()
    if return_parameter_vectors:
        return qc, betas, gammas
    else:
        return qc
