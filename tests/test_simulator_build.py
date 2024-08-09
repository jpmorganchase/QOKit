###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import pytest
import os
import glob

# from qokit.fur import get_available_simulator_names


QOKIT_PYTHON_ONLY = os.environ.get("QOKIT_PYTHON_ONLY")


def test_csim():
    assert glob.glob(f"/*/libcsim*.so") == ["libcsim.so"]


# @pytest.mark.skipif(not QOKIT_PYTHON_ONLY, reason="Fast c/c++ simulator is not installed")
# def test_simulator_c_build():
#     assert "c" in get_available_simulator_names("x")
#     assert "c" in get_available_simulator_names("xyring")
#     assert "c" in get_available_simulator_names("xycomplete")


# @pytest.mark.skipif(QOKIT_PYTHON_ONLY, reason="Fast c/c++ simulator is not installed")
# def test_simulator_python_build():
#     assert "python" in get_available_simulator_names("x")
#     assert "python" in get_available_simulator_names("xyring")
#     assert "python" in get_available_simulator_names("xycomplete")
