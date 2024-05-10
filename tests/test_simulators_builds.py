###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import pytest
import os
import numpy as np
from qokit import get_qaoa_labs_objective
from qokit.fur import get_available_simulator_names

# when fail tests only runs in Github actions. Change to true to run locally
IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == ""fails"


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test runs only in Github Actions.")
def test_simulator_c_bild():
    assert "c" in get_available_simulator_names("x")
    assert "c" in get_available_simulator_names("xyring")
    assert "c" in get_available_simulator_names("xycomplete")


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test runs only in Github Actions.")
def test_simulator_python_build():
    assert "python" in get_available_simulator_names("x")
    assert "python" in get_available_simulator_names("xyring")
    assert "python" in get_available_simulator_names("xycomplete")


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test runs only in Github Actions.")
@pytest.mark.timeout(15)
def test_simulator_timing_test():
    theta = np.random.uniform(0, 1, 280)
    f = get_qaoa_labs_objective(20, 140)
    f(theta)
    pass
