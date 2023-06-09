###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import numpy as np
import pickle
import copy
from pathlib import Path
import multiprocessing
from pytket import OpType
from pytket.extensions.quantinuum import QuantinuumBackend
from pytket.extensions.qiskit import qiskit_to_tk

import sys

sys.path.append("../code/")

from labs import get_depth_optimized_terms
from qaoa_circuit_labs import get_qaoa_circuit

max_p_dict = {
    3: 1,
    4: 2,
    5: 5,
    6: 3,
    7: 20,
}

backend = QuantinuumBackend(device_name="H1-1E")


def optimize_circuit(N):
    terms = get_depth_optimized_terms(N)
    if N in max_p_dict:
        max_p = max_p_dict[N]
    else:
        max_p = 10

    for p in range(1, max_p):
        outpath = f"data/circuit_from_qiskit_h1-1e_{N}_{p}.pickle"
        if Path(outpath).exists():
            print(f"Found precomputed pickle at {outpath}, skipping", flush=True)
            continue

        beta = np.random.uniform(0, np.pi, p)
        gamma = np.random.uniform(0, np.pi, p)
        circ = qiskit_to_tk(get_qaoa_circuit(N, terms, beta, gamma, save_statevector=False))

        compiled_circuit = backend.get_compiled_circuit(circ, optimisation_level=2)
        ZZMax_depth = compiled_circuit.depth_by_type(OpType.ZZMax)
        ZZPhase_depth = compiled_circuit.depth_by_type(OpType.ZZPhase)
        ZZMax_count = len(compiled_circuit.ops_of_type(OpType.ZZMax))
        ZZPhase_count = len(compiled_circuit.ops_of_type(OpType.ZZPhase))
        two_q_depth = ZZMax_depth + ZZPhase_depth
        two_q_count = ZZMax_count + ZZPhase_count
        row = {
            "compiled_circuit": copy.deepcopy(compiled_circuit),
            "ZZMax_depth": ZZMax_depth,
            "ZZPhase_depth": ZZPhase_depth,
            "ZZMax_count": ZZMax_count,
            "ZZPhase_count": ZZPhase_count,
            "two_q_depth": two_q_depth,
            "two_q_count": two_q_count,
            "N": N,
            "p": p,
        }
        print(f"{N}\t{p}\t{two_q_depth}\t\t{two_q_count}", flush=True)

        pickle.dump(row, open(outpath, "wb"))


if __name__ == "__main__":
    _ = backend.backend_info  # just to trigger the Quantinuum password prompt
    print("N\tp\t2q-depth\t2q-count")
    with multiprocessing.Pool(8) as p:
        p.map(optimize_circuit, range(15, 35))
