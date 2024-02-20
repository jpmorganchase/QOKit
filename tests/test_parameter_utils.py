###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import numpy as np
import pandas as pd
import pytest
from importlib_resources import files

from qokit.qaoa_objective_labs import get_qaoa_labs_objective
from qokit.fur import get_available_simulator_names
from qokit.parameter_utils import from_fourier_basis, to_fourier_basis, extrapolate_parameters_in_fourier_basis, convert_to_gamma_beta

simulators_to_run = get_available_simulator_names("x") + ["qiskit"]


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
    u, v = to_fourier_basis(gamma, beta)
    gamma2, beta2 = from_fourier_basis(*extrapolate_parameters_in_fourier_basis(u, v, p + 1, 1))
    gamma3, beta3 = from_fourier_basis(u, v)

    e1 = f_gammabeta(gamma, beta)
    e2 = f_gammabeta(gamma2, beta2)
    e3 = f_gammabeta(gamma3, beta3)
    e4 = f_uv(u, v)

    assert np.isclose(e1, e3)
    assert np.isclose(e1, e4)
    assert e2 < e1
