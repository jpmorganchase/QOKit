###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import numpy as np
import pandas as pd
from qiskit_aer import AerSimulator
from qiskit import transpile
from pathlib import Path
from qokit.utils import precompute_energies, obj_from_statevector, get_ramp, reverse_array_index_bit_order
from qokit.qaoa_circuit_labs import (
    get_parameterized_qaoa_circuit,
    get_qaoa_circuit,
)
from qokit.qaoa_circuit import get_qaoa_circuit_from_terms
from qokit.labs import negative_merit_factor_from_bitstring, get_terms_offset
from qokit.utils import (
    precompute_energies,
    obj_from_statevector,
    get_ramp,
)
from qokit.fur.diagonal_precomputation import precompute_vectorized_cpu_parallel
from qokit.qaoa_objective_labs import get_qaoa_labs_objective
from qokit import parameter_utils

test_qaoa_qiskit_folder = Path(__file__).parent


def test_qaoa_parameterization():
    N = 7
    p = 4

    df = pd.read_json(
        Path(test_qaoa_qiskit_folder, "../qokit/assets/best_LABS_QAOA_parameters_wrt_MF.json"),
        orient="index",
    )
    row = df[(df["N"] == N) & (df["p"] == p)].squeeze()

    qc = get_qaoa_circuit(N, row["gamma"], row["beta"])
    backend = AerSimulator(method="statevector")
    sv = np.asarray(backend.run(qc).result().get_statevector())

    assert np.isclose(
        obj_from_statevector(sv, negative_merit_factor_from_bitstring),
        -row["merit factor"],
    )

    assert np.isclose(
        get_qaoa_labs_objective(N, p)(np.hstack([row["gamma"], row["beta"]])),
        -row["merit factor"],
    )

    assert np.isclose(
        get_qaoa_labs_objective(N, p, parameterization="gamma beta")(row["gamma"], row["beta"]),
        -row["merit factor"],
    )

    precomputed_energies = precompute_energies(negative_merit_factor_from_bitstring, N)
    assert np.isclose(
        get_qaoa_labs_objective(N, p, precomputed_energies)(np.hstack([row["gamma"], row["beta"]])),
        -row["merit factor"],
    )


def test_parameterized_circuit():
    N = 10
    p = 50
    ramp = get_ramp(0.1663, p)

    backend = AerSimulator(method="statevector")

    qc_param = get_parameterized_qaoa_circuit(N, p)
    qc1 = qc_param.assign_parameters(np.hstack([ramp["beta"], ramp["gamma"]]))
    f1 = obj_from_statevector(
        np.asarray(backend.run(qc1).result().get_statevector()),
        negative_merit_factor_from_bitstring,
    )

    qc2 = get_qaoa_circuit(N, ramp["gamma"], ramp["beta"])
    f2 = obj_from_statevector(
        np.asarray(backend.run(qc2).result().get_statevector()),
        negative_merit_factor_from_bitstring,
    )

    f3 = get_qaoa_labs_objective(N, p, parameterization="gamma beta")(ramp["gamma"], ramp["beta"])

    assert np.isclose(f1, f2)
    assert np.isclose(f1, f3)


def test_labs_circuit_from_terms_overlap():
    """Test that building a LABS QAOA circuit via the general get_qaoa_circuit_from_terms
    and running it on Aer produces the expected overlap with optimal bitstrings."""
    N = 6
    p = 1

    gamma, beta = parameter_utils.get_best_known_parameters_for_LABS_wrt_overlap_for_p(N, p)
    terms, offset = get_terms_offset(N)

    qc = get_qaoa_circuit_from_terms(N, terms, gamma, beta)
    backend = AerSimulator(method="statevector")
    sv = reverse_array_index_bit_order(np.array(backend.run(transpile(qc, backend)).result().get_statevector()))

    diag = precompute_vectorized_cpu_parallel(terms, offset, N)
    probs = np.abs(sv) ** 2
    overlap = probs[diag == diag.min()].sum()

    # Cross-check against stored expected overlap
    params_df = parameter_utils.get_best_known_parameters_for_LABS_wrt_overlap(N)
    expected_overlap = params_df[params_df["p"] == p].squeeze()["overlap"]
    assert np.isclose(overlap, expected_overlap, atol=1e-4), f"overlap {overlap} != expected {expected_overlap}"
