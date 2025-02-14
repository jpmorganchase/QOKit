###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
# QAOA circuit for S_k

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from typing import Sequence
from .qaoa_circuit_utils import _get_qaoa_circuit, _get_parameterized_qaoa_circuit


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
    return _get_qaoa_circuit(J=J, gammas=gammas, betas=betas, save_statevector=save_statevector, qr=qr, cr=cr)


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
    return _get_parameterized_qaoa_circuit(J=J, p=p, save_statevector=save_statevector, qr=qr, cr=cr, return_parameter_vectors=return_parameter_vectors)
