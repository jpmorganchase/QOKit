###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import numpy as np
import pandas as pd
import pytest
import unittest
from importlib_resources import files

from qokit.qaoa_objective_labs import get_qaoa_labs_objective
from qokit.fur import get_available_simulator_names
from qokit.parameter_utils import to_basis, from_basis, extrapolate_parameters_in_fourier_basis, convert_to_gamma_beta

simulators_to_run = get_available_simulator_names("x") + ["qiskit"]


def test_to_and_from_basis():
    gamma = [0.5, 0.7, 0.8]
    beta = [0.3, 0.4, 0.6]
    bases = ["fourier", "chebyshev", "legendre", "hermite", "laguerre"]

    for nc in [3, 4, 5]:
        # nc >= len(gamma) for testing non-fourier basis
        for basis in bases:
            u, v = to_basis(gamma, beta, nc, basis=basis)
            g, b = from_basis(u, v, p=len(gamma), basis=basis)
            assert np.allclose(gamma, g, rtol=1e-3)
            assert np.allclose(beta, b, rtol=1e-3)


def from_fourier_basis_old(u, v):
    """Convert u,v parameterizing QAOA in the Fourier basis
    to beta, gamma in standard parameterization
    following https://arxiv.org/abs/1812.01041
    Baseline version implemented by hardcoding the formula
    Only used for tests

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
    return gamma, beta


def to_fourier_basis_old(gamma, beta):
    """Convert gamma,beta standard parameterizing QAOA to the Fourier basis
    of u, v in standard parameterization
    following https://arxiv.org/abs/1812.01041
    Baseline version implemented by hardcoding the formula
    Only used for tests

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


class TestQAOAFourier(unittest.TestCase):

    def setUp(self):
        # Set up some example data for testing
        self.lengths = [100, 200, 500]
        self.test_cases = []
        for length in self.lengths:
            u = np.random.rand(length)
            v = np.random.rand(length)
            gamma, beta = from_basis(u, v, p=None, basis="fourier")
            gamma_old, beta_old = from_fourier_basis_old(u, v)
            self.test_cases.append((u, v, gamma, beta, gamma_old, beta_old))

    def test_from_fourier_basis(self):
        for u, v, gamma, beta, gamma_old, beta_old in self.test_cases:
            gamma_new, beta_new = from_basis(u, v, p=None, basis="fourier")
            self.assertTrue(np.allclose(gamma_new, gamma_old))
            self.assertTrue(np.allclose(beta_new, beta_old))

    def test_to_fourier_basis(self):
        for u, v, gamma, beta, gamma_old, beta_old in self.test_cases:
            u_new, v_new = to_basis(gamma, beta, basis="fourier")
            self.assertTrue(np.allclose(u_new, u))
            self.assertTrue(np.allclose(v_new, v))

    def test_from_fourier_basis_old(self):
        for u, v, gamma, beta, gamma_old, beta_old in self.test_cases:
            gamma_old_new, beta_old_new = from_fourier_basis_old(u, v)
            self.assertTrue(np.allclose(gamma_old_new, gamma))
            self.assertTrue(np.allclose(beta_old_new, beta))

    def test_to_fourier_basis_old(self):
        for u, v, gamma, beta, gamma_old, beta_old in self.test_cases:
            u_old_new, v_old_new = to_fourier_basis_old(gamma_old, beta_old)
            self.assertTrue(np.allclose(u_old_new, u))
            self.assertTrue(np.allclose(v_old_new, v))

    def test_round_trip_conversion(self):
        for u, v, gamma, beta, gamma_old, beta_old in self.test_cases:
            # Test round-trip conversion for new methods
            u_new, v_new = to_basis(gamma, beta, basis="fourier")
            gamma_new, beta_new = from_basis(u_new, v_new, p=None, basis="fourier")
            self.assertTrue(np.allclose(gamma_new, gamma))
            self.assertTrue(np.allclose(beta_new, beta))

            # Test round-trip conversion for old methods
            u_old_new, v_old_new = to_fourier_basis_old(gamma_old, beta_old)
            gamma_old_new, beta_old_new = from_fourier_basis_old(u_old_new, v_old_new)
            self.assertTrue(np.allclose(gamma_old_new, gamma_old))
            self.assertTrue(np.allclose(beta_old_new, beta_old))


@pytest.mark.parametrize("simulator", simulators_to_run)
def test_extrapolation(simulator):
    df = pd.read_json(str(files("qokit.assets").joinpath("best_LABS_QAOA_parameters_wrt_MF.json")), orient="index")
    N = 15
    p = 50
    row = df[(df["N"] == N) & (df["p"] == p)].squeeze()
    beta = row["beta"]
    gamma = row["gamma"]

    f_gammabeta = get_qaoa_labs_objective(N, p, parameterization="gamma beta")
    f_uv = get_qaoa_labs_objective(N, p, parameterization="u v")
    u, v = to_basis(gamma, beta, num_coeffs=None, basis="fourier")
    gamma2, beta2 = from_basis(*extrapolate_parameters_in_fourier_basis(u, v, p + 1), p=None, basis="fourier")
    gamma3, beta3 = from_basis(u, v, p=None, basis="fourier")

    e1 = f_gammabeta(gamma, beta)
    e2 = f_gammabeta(gamma2, beta2)
    e3 = f_gammabeta(gamma3, beta3)
    e4 = f_uv(u, v)

    assert np.isclose(e1, e3)
    assert np.isclose(e1, e4)
    assert e2 < e1


def test_convert_to_gamma_beta():
    p = 10
    gamma = np.random.uniform(0, 1, p)
    beta = np.random.uniform(0, 1, p)

    u, v = to_basis(gamma, beta, num_coeffs=None, basis="fourier")

    gamma2, beta2 = convert_to_gamma_beta(u, v, parameterization="u v")
    gamma3, beta3 = convert_to_gamma_beta(np.hstack([gamma, beta]), parameterization="theta")

    assert np.allclose(gamma, gamma2)
    assert np.allclose(gamma, gamma3)
    assert np.allclose(beta, beta2)
    assert np.allclose(beta, beta3)
