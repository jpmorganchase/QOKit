###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import pytest
import os
import numpy as np
from qokit import get_qaoa_labs_objective
from qokit.fur import get_available_simulator_names

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
PYTHON_ONLY = os.environ.get("QOKIT_PYTHON_ONLY") == "true"


@pytest.mark.skipif(PYTHON_ONLY, reason="Fast c/c++ simulator is not installed")
def test_simulator_c_build():
    assert "c" in get_available_simulator_names("x")
    assert "c" in get_available_simulator_names("xyring")
    assert "c" in get_available_simulator_names("xycomplete")


@pytest.mark.skipif(not PYTHON_ONLY, reason="Fast c/c++ simulator should be installed")
def test_simulator_lack_of_c_build():
    assert "c" not in get_available_simulator_names("x")
    assert "c" not in get_available_simulator_names("xyring")
    assert "c" not in get_available_simulator_names("xycomplete")


def test_simulator_python_build():
    assert "python" in get_available_simulator_names("x")
    assert "python" in get_available_simulator_names("xyring")
    assert "python" in get_available_simulator_names("xycomplete")


@pytest.mark.skipif(PYTHON_ONLY, reason="Fast c/c++ simulator is not installed")
@pytest.mark.timeout(10)
def test_simulator_timing():
    theta = np.random.uniform(0, 1, 280)
    f = get_qaoa_labs_objective(20, 140)
    f(theta)
