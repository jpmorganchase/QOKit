###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
# Utilities for parameter initialization

from __future__ import annotations
import numpy as np
from pathlib import Path
import pandas as pd
from importlib_resources import files
from enum import Enum
from typing import Callable


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
        beta, gamma = from_fourier_basis(u, v)
    elif parameterization == QAOAParameterization.GAMMA_BETA:
        assert len(args) == 2, "gamma beta parameterization requires two arguments"
        gamma, beta = args
    elif parameterization == QAOAParameterization.U_V:
        assert len(args) == 2, "u v parameterization requires two arguments"
        u, v = args
        beta, gamma = from_fourier_basis(u, v)
    else:
        raise ValueError("Invalid parameterization")
    return gamma, beta


def get_sk_gamma_beta(p, parameterization: QAOAParameterization | str = "gamma beta"):
    """
    Load the look-up table for initial points from
    https://arxiv.org/pdf/2110.14206.pdf
    """
    if p == 1:
        gamma = np.array([0.5])
        beta = np.array([np.pi / 8])
    elif p == 2:
        gamma = np.array([0.3817, 0.6655])
        beta = np.array([0.4960, 0.2690])
    elif p == 3:
        gamma = np.array([0.3297, 0.5688, 0.6406])
        beta = np.array([0.5500, 0.3675, 0.2109])
    elif p == 4:
        gamma = np.array([0.2949, 0.5144, 0.5586, 0.6429])
        beta = np.array([0.5710, 0.4176, 0.3028, 0.1729])
    elif p == 5:
        gamma = np.array([0.2705, 0.4804, 0.5074, 0.5646, 0.6397])
        beta = np.array([0.5899, 0.4492, 0.3559, 0.2643, 0.1486])
    elif p == 6:
        gamma = np.array([0.2528, 0.4531, 0.4750, 0.5146, 0.5650, 0.6392])
        beta = np.array([0.6004, 0.4670, 0.3880, 0.3176, 0.2325, 0.1291])
    elif p == 7:
        gamma = np.array([0.2383, 0.4327, 0.4516, 0.4830, 0.5147, 0.5686, 0.6393])
        beta = np.array([0.6085, 0.4810, 0.4090, 0.3534, 0.2857, 0.2080, 0.1146])
    elif p == 8:
        gamma = np.array([0.2268, 0.4162, 0.4332, 0.4608, 0.4818, 0.5179, 0.5717, 0.6393])
        beta = np.array([0.6151, 0.4906, 0.4244, 0.3780, 0.3224, 0.2606, 0.1884, 0.1030])
    elif p == 9:
        gamma = np.array([0.2172, 0.4020, 0.4187, 0.4438, 0.4592, 0.4838, 0.5212, 0.5754, 0.6398])
        beta = np.array([0.6196, 0.4973, 0.4354, 0.3956, 0.3481, 0.2973, 0.2390, 0.1717, 0.0934])
    elif p == 10:
        gamma = np.array([0.2089, 0.3902, 0.4066, 0.4305, 0.4423, 0.4604, 0.4858, 0.5256, 0.5789, 0.6402])
        beta = np.array([0.6235, 0.5029, 0.4437, 0.4092, 0.3673, 0.3246, 0.2758, 0.2208, 0.1578, 0.0855])
    elif p == 11:
        gamma = np.array([0.2019, 0.3799, 0.3963, 0.4196, 0.4291, 0.4431, 0.4611, 0.4895, 0.5299, 0.5821, 0.6406])
        beta = np.array([0.6268, 0.5070, 0.4502, 0.4195, 0.3822, 0.3451, 0.3036, 0.2571, 0.2051, 0.1459, 0.0788])
    elif p == 12:
        gamma = np.array([0.1958, 0.3708, 0.3875, 0.4103, 0.4185, 0.4297, 0.4430, 0.4639, 0.4933, 0.5343, 0.5851, 0.6410])
        beta = np.array([0.6293, 0.5103, 0.4553, 0.4275, 0.3937, 0.3612, 0.3248, 0.2849, 0.2406, 0.1913, 0.1356, 0.0731])
    elif p == 13:
        gamma = np.array([0.1903, 0.3627, 0.3797, 0.4024, 0.4096, 0.4191, 0.4290, 0.4450, 0.4668, 0.4975, 0.5385, 0.5878, 0.6414])
        beta = np.array([0.6315, 0.5130, 0.4593, 0.4340, 0.4028, 0.3740, 0.3417, 0.3068, 0.2684, 0.2260, 0.1792, 0.1266, 0.0681])
    elif p == 14:
        gamma = np.array([0.1855, 0.3555, 0.3728, 0.3954, 0.4020, 0.4103, 0.4179, 0.4304, 0.4471, 0.4703, 0.5017, 0.5425, 0.5902, 0.6418])
        beta = np.array([0.6334, 0.5152, 0.4627, 0.4392, 0.4103, 0.3843, 0.3554, 0.3243, 0.2906, 0.2535, 0.2131, 0.1685, 0.1188, 0.0638])
    elif p == 15:
        gamma = np.array([0.1811, 0.3489, 0.3667, 0.3893, 0.3954, 0.4028, 0.4088, 0.4189, 0.4318, 0.4501, 0.4740, 0.5058, 0.5462, 0.5924, 0.6422])
        beta = np.array([0.6349, 0.5169, 0.4655, 0.4434, 0.4163, 0.3927, 0.3664, 0.3387, 0.3086, 0.2758, 0.2402, 0.2015, 0.1589, 0.1118, 0.0600])
    elif p == 16:
        gamma = np.array([0.1771, 0.3430, 0.3612, 0.3838, 0.3896, 0.3964, 0.4011, 0.4095, 0.4197, 0.4343, 0.4532, 0.4778, 0.5099, 0.5497, 0.5944, 0.6425])
        beta = np.array([0.6363, 0.5184, 0.4678, 0.4469, 0.4213, 0.3996, 0.3756, 0.3505, 0.3234, 0.2940, 0.2624, 0.2281, 0.1910, 0.1504, 0.1056, 0.0566])
    elif p == 17:
        gamma = np.array(
            [0.1735, 0.3376, 0.3562, 0.3789, 0.3844, 0.3907, 0.3946, 0.4016, 0.4099, 0.4217, 0.4370, 0.4565, 0.4816, 0.5138, 0.5530, 0.5962, 0.6429]
        )
        beta = np.array(
            [0.6375, 0.5197, 0.4697, 0.4499, 0.4255, 0.4054, 0.3832, 0.3603, 0.3358, 0.3092, 0.2807, 0.2501, 0.2171, 0.1816, 0.1426, 0.1001, 0.0536]
        )
    else:
        raise ValueError(f"p={p} not supported, try lower p")
    parameterization = QAOAParameterization(parameterization)
    if parameterization == QAOAParameterization.THETA:
        return np.concatenate((4 * gamma, beta), axis=0)
    elif parameterization == QAOAParameterization.GAMMA_BETA:
        return 4 * gamma, beta


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
    df = pd.read_json(str(files("qokit.assets.maxcut_datasets").joinpath("fixed_angles_for_regular_graphs.json")), orient="index")

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
