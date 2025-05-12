import pandas as pd
import numpy as np
from qokit.qaoa_objective_labs import get_qaoa_labs_objective
from qokit.labs import get_terms
import subprocess
from qokit.fur.mpi_nbcuda.qaoa_simulator import mpi_available
from qokit.fur import get_available_simulator_names

import subprocess

import os, sys
import pytest

simulators = get_available_simulator_names("x")
result = subprocess.run(["nvidia-smi", "nvlink", "-cBridge"], capture_output=True, text=True, check=True)

output_lines = result.stdout.strip().split("\n")

is_nvlink = output_lines is not [""] and len(output_lines) > 1
is_mpi_available = mpi_available()
is_gpumpi = "gpumpi" in simulators
n_gpumpi = len(output_lines)
is_gpu = "gpu" in simulators


@pytest.mark.skipif(not is_nvlink, reason="NVLINK not available.")
def test_gpumpi(n_procs=n_gpumpi):
    result = subprocess.run(["mpirun", "-np", str(n_procs), "python", "tests/gpumpi_tests.py"], capture_output=True, text=True, check=True)
    output_lines = result.stdout.strip().split("\n")
    output = [item == "True" for item in output_lines]
    assert all(output)


@pytest.mark.skipif(not is_gpu, reason="GPU simulator not available.")
def test_gpu(n_procs=1):
    result = subprocess.run(["mpirun", "-np", str(n_procs), "python", "tests/gpumpi_tests.py"], capture_output=True, text=True, check=True)
    output_lines = result.stdout.strip().split("\n")
    output = [item == "True" for item in output_lines]
    assert all(output)
