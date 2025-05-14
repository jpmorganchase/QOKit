import pytest
import pandas as pd
import numpy as np
import networkx as nx
from functools import partial
import subprocess

from qokit.qaoa_objective_labs import get_qaoa_labs_objective
from qokit.fur.mpi_nbcuda.qaoa_simulator import mpi_available
from qokit.fur import get_available_simulator_names
from qokit.maxcut import maxcut_obj, get_adjacency_matrix
from qokit.qaoa_objective_maxcut import get_qaoa_maxcut_objective
from qokit.sk import sk_obj, get_random_J
from qokit.qaoa_objective_sk import get_qaoa_sk_objective
from qokit.utils import precompute_energies
from qokit.parameter_utils import get_sk_gamma_beta, get_fixed_gamma_beta


simulators = get_available_simulator_names("x")

if "gpu" in simulators:
    result = subprocess.run(["nvidia-smi", "nvlink", "-cBridge"], capture_output=True, text=True, check=True)
    output_lines = result.stdout.strip().split("\n")
    is_nvlink = output_lines is not [""] and len(output_lines) > 1
    n_gpumpi = len(output_lines)
else:
    is_nvlink = False
    n_gpumpi = 0


is_mpi_available = mpi_available()
is_gpumpi = "gpumpi" in simulators
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
def test_gpu_labs(N=16, p=4, seed=1):
    df = pd.read_json("./qokit/assets/QAOA_with_fixed_parameters_p_opt.json", orient="index")
    row = df[(df["N"] == N) & (df["p"] == p)]
    beta = row["beta"].values[0]
    gamma = row["gamma"].values[0]
    overlap_trans_labs = float(row["overlap transferred"].values[0])

    f_gpu_labs = get_qaoa_labs_objective(N, p, simulator="gpu", objective="overlap", parameterization="gamma beta")
    overlap_trans_computed_labs = 1 - f_gpu_labs(gamma, beta)
    assert np.isclose(overlap_trans_computed_labs, overlap_trans_labs)


@pytest.mark.skipif(not is_gpu, reason="GPU simulator not available.")
def test_gpu_maxcut(N=16, p=4, seed=1):
    d = 3
    G = nx.random_regular_graph(d, N, seed=seed)
    obj_maxcut = partial(maxcut_obj, w=get_adjacency_matrix(G))
    precomputed_energies = precompute_energies(obj_maxcut, N)
    gamma, beta, AR = get_fixed_gamma_beta(d, p, return_AR=True)

    o1_c_maxcut = get_qaoa_maxcut_objective(N, p, precomputed_cuts=precomputed_energies, simulator="c", parameterization="gamma beta", objective="overlap")(
        gamma, beta
    )
    o1_gpu = get_qaoa_maxcut_objective(N, p, precomputed_cuts=precomputed_energies, simulator="gpu", parameterization="gamma beta", objective="overlap")(
        gamma, beta
    )
    o2_gpu = get_qaoa_maxcut_objective(N, p, G=G, simulator="gpu", parameterization="gamma beta", objective="overlap")(gamma, beta)
    assert np.all([np.isclose(o1_c_maxcut, np.real(o1_gpu)), np.isclose(o1_c_maxcut, np.real(o2_gpu))])


@pytest.mark.skipif(not is_gpu, reason="GPU simulator not available.")
def test_gpu_sk(N=16, p=4, seed=1):
    J = get_random_J(N=N)
    obj_sk = partial(sk_obj, J=J)
    precomputed_energies = precompute_energies(obj_sk, N)
    gamma, beta = get_sk_gamma_beta(p)

    o1_c_sk = get_qaoa_sk_objective(N, p, J=J, precomputed_energies=precomputed_energies, simulator="c", parameterization="gamma beta", objective="overlap")(
        gamma, beta
    )
    o1_gpu = get_qaoa_sk_objective(N, p, J=J, precomputed_energies=precomputed_energies, simulator="gpu", parameterization="gamma beta", objective="overlap")(
        gamma, beta
    )
    o2_gpu = get_qaoa_sk_objective(N, p, J=J, simulator="gpu", parameterization="gamma beta", objective="overlap")(gamma, beta)

    assert np.all([np.isclose(o1_c_sk, np.real(o1_gpu)), np.isclose(o1_c_sk, np.real(o2_gpu))])
