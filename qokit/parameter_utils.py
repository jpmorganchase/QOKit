###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
# Utilities for parameter initialization

import os
import sys
import time
import numpy as np
import pandas as pd
from importlib_resources import files
from enum import Enum
from functools import cache
from scipy.fft import dct, dst, idct, idst


def to_basis(gamma, beta, num_coeffs=None, basis="fourier"):
    """Convert gamma,beta angles in standard parameterizing QAOA to a basis of functions

    Parameters
    ----------
    gamma : list-like
    beta : list-like
    num_coeffs : int
    basis : string
        QAOA parameters in standard basis
    Returns
    -------
    u, v : np.array
        QAOA parameters in given basis
    """
    if basis == "fourier":
        u = 2 * dst(gamma, type=4, norm="forward")  # difference of 2 due to normalization of dst
        v = 2 * dct(beta, type=4, norm="forward")  # difference of 2 due to normalization of dct
    else:
        assert num_coeffs is not None
        fit_interval = np.linspace(-1, 1, len(gamma))
        if basis == "chebyshev":
            u = np.polynomial.chebyshev.chebfit(fit_interval, gamma, deg=num_coeffs - 1)  # offset of 1 due to fitting convention
            v = np.polynomial.chebyshev.chebfit(fit_interval, beta, deg=num_coeffs - 1)
        elif basis == "hermite":
            u = np.polynomial.hermite.hermfit(fit_interval, gamma, deg=num_coeffs - 1)
            v = np.polynomial.hermite.hermfit(fit_interval, beta, deg=num_coeffs - 1)
        elif basis == "legendre":
            u = np.polynomial.legendre.legfit(fit_interval, gamma, deg=num_coeffs - 1)
            v = np.polynomial.legendre.legfit(fit_interval, beta, deg=num_coeffs - 1)
        elif basis == "laguerre":
            u = np.polynomial.laguerre.lagfit(fit_interval, gamma, deg=num_coeffs - 1)
            v = np.polynomial.laguerre.lagfit(fit_interval, beta, deg=num_coeffs - 1)

    return u, v


def from_basis(u, v, p=None, basis="fourier"):
    """Convert u,v in a given basis of functions
    to gamma, beta angles of QAOA schedule

    Parameters
    ----------
    u : list-like
    v : list-like
    p : int, the number of coefficients
    basis : string

    Returns
    -------
    gamma, beta : np.array
        QAOA angles parameters in standard parameterization
    """
    assert len(u) == len(v)

    if basis == "fourier":
        if p is None:
            p = len(u)
        if p < len(u):
            raise Exception("p must be greater or equal the length of u and v ")

        u_padded = np.zeros(p)
        v_padded = np.zeros(p)
        u_padded[: len(u)] = u
        v_padded[: len(v)] = v
        u_padded[len(u) :] = 0
        v_padded[len(v) :] = 0

        gamma = 0.5 * idst(u_padded, type=4, norm="forward")  # difference of 1/2 due to normalization of idst
        beta = 0.5 * idct(v_padded, type=4, norm="forward")  # difference of 1/2 due to normalization of idct
    else:
        assert p is not None
        fit_interval = np.linspace(-1, 1, p)

    if basis == "chebyshev":
        gamma = np.polynomial.chebyshev.chebval(fit_interval, u)
        beta = np.polynomial.chebyshev.chebval(fit_interval, v)
    elif basis == "hermite":
        gamma = np.polynomial.hermite.hermval(fit_interval, u)
        beta = np.polynomial.hermite.hermval(fit_interval, v)
    elif basis == "legendre":
        gamma = np.polynomial.legendre.legval(fit_interval, u)
        beta = np.polynomial.legendre.legval(fit_interval, v)
    elif basis == "laguerre":
        gamma = np.polynomial.laguerre.lagval(fit_interval, u)
        beta = np.polynomial.laguerre.lagval(fit_interval, v)

    return gamma, beta


def extrapolate_parameters_in_fourier_basis(u, v, p):
    """Extrapolate the parameters u, v to p
    Parameters
    ----------
    u : list-like
    v : list-like
        QAOA parameters in Fourier basis
    p : int
        QAOA depth
    Returns
    -------
    u, v : np.array
        QAOA parameters in Fourier basis
        for depth p.
    """

    u_next = np.zeros(p)
    v_next = np.zeros(p)

    p_k = min(p, len(u))
    u_next[:p_k] = u[:p_k]
    v_next[:p_k] = v[:p_k]

    return u_next, v_next


class QAOAParameterization(Enum):
    """
    Enum class to specify the parameterization of the QAOA parameters
    """

    THETA = "theta"
    GAMMA_BETA = "gamma beta"
    FREQ = "freq"
    U_V = "u v"


def convert_to_gamma_beta(*args, parameterization: QAOAParameterization | str):
    """
    Convert QAOA parameters to gamma, beta parameterization
    """
    parameterization = QAOAParameterization(parameterization)
    if parameterization == QAOAParameterization.THETA:
        assert len(args) == 1, "theta parameterization requires a single argument"
        theta = args[0]
        p = int(len(theta) / 2)
        gamma = theta[:p]
        beta = theta[p:]
    elif parameterization == QAOAParameterization.FREQ:
        assert len(args) == 1, "freq parameterization requires two arguments"
        freq = args[0]
        p = int(len(freq) / 2)
        u = freq[:p]
        v = freq[p:]
        gamma, beta = from_basis(u, v, p=None, basis="fourier")
    elif parameterization == QAOAParameterization.GAMMA_BETA:
        assert len(args) == 2, "gamma beta parameterization requires two arguments"
        gamma, beta = args
    elif parameterization == QAOAParameterization.U_V:
        assert len(args) == 2, "u v parameterization requires two arguments"
        u, v = args
        gamma, beta = from_basis(u, v, p=None, basis="fourier")
    else:
        raise ValueError("Invalid parameterization")
    return gamma, beta


@cache
def _get_sk_gamma_beta_from_file():
    """
    Caches the dataframe after the first call to load JSon, subsequent calls will get from cache and save I/O

    Parameters
    ----------
    None

    Returns
    -------
    df: Pandas Dataframe
    """
    return pd.read_json(str(files("qokit.assets").joinpath("best_SK_QAOA_parameters.json")), orient="index")


def _get_sk_gamma_beta(p):
    """
    Returns the parameters for QAOA for infinite-sized SK model

    Parameters
    ----------
    p : int
        QAOA depth

    Returns
    -------
    gamma, beta : (list, list)
        Parameters as two separate lists in a tuple
    """
    df = _get_sk_gamma_beta_from_file()
    row = df[(df["p"] == p)]

    # If the angles aren't in the database, we extrapolate from nearest p
    if len(row) != 1:
        p_list = np.array(df["p"].keys())
        print(f"p_list = {p_list}")
        p_closest = p_list[np.argmin(np.abs(p_list - p))]
        row = df[(df["p"] == p_closest)].squeeze()
        print(f"Extrapolating from p={p_closest}")
        gamma, beta = np.array(row["gammas"]), np.array(row["betas"])
        u, v = to_basis(gamma, beta)
        u_next, v_next = extrapolate_parameters_in_fourier_basis(u, v, p)
        gamma, beta = from_basis(u_next, v_next)
        # raise ValueError(f"p={p} not supported, try lower p")
        return np.array(gamma), np.array(beta)
    else:
        row = row.squeeze()
        return np.array(row["gammas"]), np.array(row["betas"])


def get_sk_gamma_beta(p, parameterization: QAOAParameterization | str = "gamma beta"):
    """
    Load the look-up table for initial points from json file
    """
    gamma, beta = _get_sk_gamma_beta(p)
    parameterization = QAOAParameterization(parameterization)
    if parameterization == QAOAParameterization.THETA:
        return np.concatenate((-2 * gamma, beta), axis=0)
    elif parameterization == QAOAParameterization.GAMMA_BETA:
        return -2 * gamma, beta


@cache
def _get_gamma_beta_from_file():
    """
    Caches the dataframe after the first call to load JSon, subsequent calls will get from cache and save I/O

    Parameters
    ----------
    None

    Returns
    -------
    df: Pandas Dataframe
    """
    return pd.read_json(str(files("qokit.assets.maxcut_datasets").joinpath("fixed_angles_for_regular_graphs.json")), orient="index")


def get_fixed_gamma_beta(d, p, return_AR=False):
    """
    Returns the parameters for QAOA for MaxCut on regular graphs from arXiv:2107.00677

    Parameters
    ----------
    d : int
        Degree of the graph
    p : int
        QAOA depth
    return_AR : bool
        return the guaranteed approximation ratio

    Returns
    -------
    gamma, beta : (list, list)
        Parameters as two separate lists in a tuple
    AR : float
        Only returned is flag return_AR is raised
    """
    df = _get_gamma_beta_from_file()
    row = df[(df["d"] == d) & (df["p"] == p)]
    if len(row) != 1:
        raise ValueError(f"Failed to retrieve fixed angles for d={d}, p={p}")
    row = row.squeeze()
    if return_AR:
        return row["gamma"], row["beta"], row["AR"]
    else:
        return row["gamma"], row["beta"]


def get_best_known_parameters_for_LABS_wrt_overlap(N: int) -> pd.DataFrame:
    """
    Loads best known LABS QAOA parameters with respect to overlap with ground state.
    Note that these parameters may be different from optimal
    parameters w.r.t expectation value of the cost hamiltonian.

    The scaling of parameters with N is taken from arXiv:1411.4028.


    Parameters
    ----------
    N : int
        Number of qubits

    Returns
    -------
    df : pd.DataFrame  DataFrame with all known values

    """
    df = pd.read_json(str(files("qokit.assets").joinpath("best_LABS_QAOA_parameters_wrt_overlap.json")), orient="index")
    df = df[df["N"] == N]
    return df


def get_best_known_parameters_for_LABS_wrt_overlap_for_p(N: int, p: int) -> tuple[list[float], list[float]]:
    """
    Loads best known LABS QAOA parameters with respect to overlap with ground state.
    Note that these parameters may be different from optimal
    parameters w.r.t expectation value of the cost hamiltonian.

    Parameters
    ----------
    N : int
        Number of qubits.
    p : int
        Number of QAOA layers.

    Returns
    -------
    gamma, beta : list[float]  QAOA parameters for fixed p if specified
    """
    df = get_best_known_parameters_for_LABS_wrt_overlap(N)
    if p > int(df["p"].max()):
        raise ValueError(f"QAOA values for p={p} is not known for N={N}")
    row = df[df["p"] == p].squeeze()
    gamma = [float(x) for x in row["gamma"]]
    beta = [float(x) for x in row["beta"]]
    return gamma, beta
