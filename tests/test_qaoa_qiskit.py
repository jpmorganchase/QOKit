###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import numpy as np
import pandas as pd
from qiskit.providers.aer import AerSimulator
from pathlib import Path
from qokit.utils import precompute_energies, obj_from_statevector, get_ramp
from qokit.qaoa_circuit_labs import (
    get_parameterized_qaoa_circuit,
    get_qaoa_circuit,
)
from qokit.labs import (
    get_energy_term_indices,
    negative_merit_factor_from_bitstring,
)
from qokit.utils import (
    precompute_energies,
    obj_from_statevector,
    get_ramp,
)

from qokit.qaoa_objective_labs import get_qaoa_labs_objective

test_qaoa_qiskit_folder = Path(__file__).parent


def test_qaoa_parameterization():
    N = 7
    p = 4

    df = pd.read_json(
        Path(test_qaoa_qiskit_folder, "../qokit/assets/best_LABS_QAOA_parameters_wrt_MF.json"),
        orient="index",
    )
    row = df[(df["N"] == N) & (df["p"] == p)].squeeze()

    terms_ix, offset = get_energy_term_indices(N)

    qc = get_qaoa_circuit(N, terms_ix, row["beta"], row["gamma"])
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
    terms, offset = get_energy_term_indices(N)

    backend = AerSimulator(method="statevector")

    qc_param = get_parameterized_qaoa_circuit(N, terms, p)
    qc1 = qc_param.bind_parameters(np.hstack([ramp["beta"], ramp["gamma"]]))
    f1 = obj_from_statevector(
        np.asarray(backend.run(qc1).result().get_statevector()),
        negative_merit_factor_from_bitstring,
    )

    qc2 = get_qaoa_circuit(N, terms, ramp["beta"], ramp["gamma"])
    f2 = obj_from_statevector(
        np.asarray(backend.run(qc2).result().get_statevector()),
        negative_merit_factor_from_bitstring,
    )

    f3 = get_qaoa_labs_objective(N, p, parameterization="gamma beta")(ramp["gamma"], ramp["beta"])

    assert np.isclose(f1, f2)
    assert np.isclose(f1, f3)
