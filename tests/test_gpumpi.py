import pandas as pd
import numpy as np
from qokit.qaoa_objective_labs import get_qaoa_labs_objective
from qokit.labs import get_terms
import subprocess
from qokit.fur.mpi_nbcuda.qaoa_simulator import mpi_available
from qokit.fur import get_available_simulator_names, choose_simulator

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
    result = subprocess.run(["mpirun", "-np", str(n_procs), "python", "./tests/gpumpi_tests.py"], capture_output=True, text=True, check=True)
    output_lines = result.stdout.strip().split("\n")
    output = [item == "True" for item in output_lines]
    assert all(output)


@pytest.mark.skipif(not is_gpu, reason="GPU simulator not available.")
def test_gpumpi_singleprocs():
    result = subprocess.run(["python", "./tests/gpumpi_tests.py"], capture_output=True, text=True, check=True)
    output_lines = result.stdout.strip().split("\n")
    output = [item == "True" for item in output_lines]
    assert all(output)


@pytest.mark.skipif(not is_gpu, reason="GPU simulator not available.")
def test_gpu(N=22, p=4):
    df = pd.read_json("./qokit/assets/QAOA_with_fixed_parameters_p_opt.json", orient="index")
    row = df[(df["N"] == N) & (df["p"] == p)]
    beta = row["beta"].values[0]
    gamma = row["gamma"].values[0]
    overlap_trans = float(row["overlap transferred"].values[0])

    f_gpu = get_qaoa_labs_objective(N, p, simulator="gpu", objective="overlap", parameterization="gamma beta")
    overlap_trans_computed = 1 - f_gpu(gamma, beta)
    assert np.isclose(overlap_trans_computed, overlap_trans)
