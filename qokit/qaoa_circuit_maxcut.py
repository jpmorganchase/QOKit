###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
# QAOA circuit for MAXCUT

import networkx as nx
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from typing import Sequence
from .maxcut import get_maxcut_terms
from .qaoa_circuit import get_qaoa_circuit_from_terms, get_parameterized_qaoa_circuit_from_terms


def get_qaoa_circuit(G: nx.Graph, gammas: Sequence, betas: Sequence, save_statevector: bool = True, qr: QuantumRegister = None, cr: ClassicalRegister = None):
    """Generates a circuit for weighted MaxCut on graph G.
    Parameters
    ----------
    G : networkx.Graph
        Graph to solve MaxCut on
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

    terms = get_maxcut_terms(G)
    N = G.number_of_nodes()
    return get_qaoa_circuit_from_terms(N=N, terms=terms[:-1], gammas=gammas, betas=betas, save_statevector=save_statevector, qr=qr, cr=cr)


def get_parameterized_qaoa_circuit(
    G: nx.Graph, p: int, save_statevector: bool = True, qr: QuantumRegister = None, cr: ClassicalRegister = None, return_parameter_vectors: bool = False
):
    """Generates a parameterized circuit for weighted MaxCut on graph G.
    This version is recommended for long circuits

    Parameters
    ----------
    G : networkx.Graph
        Graph to solve MaxCut on
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
    terms = get_maxcut_terms(G)
    N = G.number_of_nodes()
    return get_parameterized_qaoa_circuit_from_terms(
        N=N, terms=terms[:-1], p=p, save_statevector=save_statevector, qr=qr, cr=cr, return_parameter_vectors=return_parameter_vectors
    )
