###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
# Utilities for parameter initialization

import numpy as np


def from_fourier_basis(u, v):
    """Convert u,v parameterizing QAOA in the Fourier basis
    to beta, gamma in standard parameterization
    following https://arxiv.org/abs/1812.01041
    Parameters
    ----------
    u : list-like
    v : list-like
        QAOA parameters in Fourier basis
    Returns
    -------
    beta, gamma : np.array
        QAOA parameters in standard parameterization
        (used e.g. by qaoa_qiskit.py)
    """

    assert len(u) == len(v)
    p = len(u)
    gamma = np.zeros(p)
    beta = np.zeros(p)
    for i in range(p):
        for j in range(p):
            gamma[i] += u[j] * np.sin(((j + 1) - 0.5) * ((i + 1) - 0.5) * np.pi / p)
            beta[i] += v[j] * np.cos(((j + 1) - 0.5) * ((i + 1) - 0.5) * np.pi / p)
    return beta, gamma


def to_fourier_basis(gamma, beta):
    """Convert gamma,beta standard parameterizing QAOA to the Fourier basis
    of u, v in standard parameterization
    following https://arxiv.org/abs/1812.01041
    Parameters
    ----------
    gamma : list-like
    beta : list-like
        QAOA parameters in standard basis
    Returns
    -------
    u, v : np.array
        QAOA parameters in fourier parameterization
        (used e.g. by qaoa_qiskit.py)
    """

    assert len(gamma) == len(beta)
    p = len(gamma)
    A = np.zeros((p, p))
    B = np.zeros((p, p))
    # Build matrix for linear system solving
    for i in range(p):
        for j in range(p):
            A[i][j] = np.sin(((j + 1) - 0.5) * ((i + 1) - 0.5) * np.pi / p)
            B[i][j] = np.cos(((j + 1) - 0.5) * ((i + 1) - 0.5) * np.pi / p)
    u = np.linalg.solve(A, gamma)
    v = np.linalg.solve(B, beta)
    if np.allclose(np.dot(A, u), gamma) == True & np.allclose(np.dot(B, v), beta) == True:
        return u, v
    else:
        raise ValueError("Linear solving was incorrect")


def extrapolate_parameters_in_fourier_basis(u, v, p, step_size):
    """Extrapolate the parameters u, v from p to p+step_size
    Parameters
    ----------
    u : list-like
    v : list-like
        QAOA parameters in Fourier basis
    p : int
        QAOA depth
    step_size : int
        Target QAOA depth for extrapolation
    Returns
    -------
    u, v : np.array
        QAOA parameters in Fourier basis
        for depth p+step_size
    """

    u_next = np.zeros(p)
    v_next = np.zeros(p)
    u_next[: p - step_size] = u
    v_next[: p - step_size] = v
    u_next[p - step_size :] = 0
    v_next[p - step_size :] = 0

    return u_next, v_next
