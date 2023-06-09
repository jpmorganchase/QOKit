###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from pathlib import Path
import numpy as np
import pickle
import multiprocessing
from qiskit.providers.aer import AerSimulator

import sys

sys.path.append("../code/")

from qaoa_circuit_labs import get_qaoa_circuit
from labs import (
    get_energy_term_indices,
    negative_merit_factor_from_bitstring,
    true_optimal_mf,
)
from utils import precompute_energies, obj_from_statevector, get_ramp


def generate_phase_diagram(N):
    outpath = f"data/phase_diagram_{N}_extra_p.pickle"
    if Path(outpath).exists():
        print(f"Found precomputed phase diagram at {outpath}, exiting")
        return

    terms, offset = get_energy_term_indices(N)
    precomputed_energies = precompute_energies(negative_merit_factor_from_bitstring, N)

    def f(beta, gamma):
        qc = get_qaoa_circuit(N, terms, beta, gamma)
        backend = AerSimulator(method="statevector")
        sv = backend.run(qc).result().get_statevector()
        f_theta = obj_from_statevector(
            sv,
            negative_merit_factor_from_bitstring,
            precomputed_energies=precomputed_energies,
        )
        return -f_theta / true_optimal_mf[N]

    ps = np.arange(100, 195, 5)
    deltas = np.linspace(0.01, 1, 20)[:11]

    energies = np.zeros((len(deltas), len(ps)))

    for p_idx, p in enumerate(ps):
        for delta_idx, delta in enumerate(deltas):
            angles = get_ramp(delta, p)
            energies[delta_idx, p_idx] = f(angles["beta"], angles["gamma"])

    result = {
        "ps": ps,
        "deltas": deltas,
        "energies": energies,
        "N": N,
    }
    print(f"Saving result to {outpath}")
    pickle.dump(result, open(outpath, "wb"))


if __name__ == "__main__":
    with multiprocessing.Pool(5) as p:
        p.map(generate_phase_diagram, [7, 8, 9, 10, 11])
