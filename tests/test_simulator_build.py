###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import pytest
import os
import numpy as np
from qokit import get_qaoa_labs_objective
from qokit.fur import get_available_simulator_names
import sys

# Set up QOKIT_PYTHON_ONLY in your local enviroment for Python only
PYTHON_ONLY = False if os.environ.get("QOKIT_PYTHON_ONLY") is None else os.environ.get("QOKIT_PYTHON_ONLY")


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="Fast c/c++ simulator should be installed")
def test_simulator_lack_of_c_build():
    if PYTHON_ONLY:
        assert "c" not in get_available_simulator_names("x")
        assert "c" not in get_available_simulator_names("xyring")
        assert "c" not in get_available_simulator_names("xycomplete")
    elif not PYTHON_ONLY:
        assert "c" in get_available_simulator_names("x")
        assert "c" in get_available_simulator_names("xyring")
        assert "c" in get_available_simulator_names("xycomplete")


def test_simulator_python_build():
    assert "python" in get_available_simulator_names("x")
    assert "python" in get_available_simulator_names("xyring")
    assert "python" in get_available_simulator_names("xycomplete")
