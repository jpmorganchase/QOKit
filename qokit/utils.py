###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import numpy as np
import pandas as pd
import qiskit
from itertools import product
from multiprocessing import Pool
from qiskit import QuantumCircuit
from pytket import OpType
from pytket.circuit import Circuit
from pytket.extensions.quantinuum import QuantinuumBackend
from pytket.extensions.qiskit import qiskit_to_tk
from importlib_resources import files

from pytket.passes import (
    SequencePass,
    auto_squash_pass,
    RemoveRedundancies,
    SimplifyInitial,
    FullPeepholeOptimise,
    NormaliseTK2,
    DecomposeTK2,
    CommuteThroughMultis,
)


def get_all_best_known():
    """Get both dataframes of best known parameters for LABS merged into one

    Returns
    ----------
    df : pandas.DataFrame
        Columns corresponding to results where the parameters were optimized
        with respect to merit factor have ' opt4MF' appended to them,
        and columns optimized for overlap have ' opt4overlap' appended to them
    """
    df1 = pd.read_json(
        files("qokit.assets").joinpath("best_LABS_QAOA_parameters_wrt_MF.json"),
        orient="index",
    ).drop("nseeds", axis=1)
    df2 = pd.read_json(
        files("qokit.assets").joinpath("best_LABS_QAOA_parameters_wrt_overlap.json"),
        orient="index",
    )

    df1 = df1.set_index(["N", "p"]).add_suffix(" opt4MF")
    df2 = df2.set_index(["N", "p"]).add_suffix(" opt4overlap")

    return df1.merge(df2, left_index=True, right_index=True, how="outer").reset_index()


def brute_force(obj_f, num_variables: int, minimize: bool = False, function_takes: str = "spins"):
    """Get the maximum of a function by complete enumeration
    Returns the maximum value and the extremizing bit string
    """
    if minimize:
        best_cost_brute = float("inf")
        compare = lambda x, y: x < y
    else:
        best_cost_brute = float("-inf")
        compare = lambda x, y: x > y
    bit_strings = (((np.array(range(2**num_variables))[:, None] & (1 << np.arange(num_variables)))) > 0).astype(int)
    for x in bit_strings:
        if function_takes == "spins":
            cost = obj_f(1 - 2 * np.array(x))
        elif function_takes == "bits":
            cost = obj_f(np.array(x))
        if compare(cost, best_cost_brute):
            best_cost_brute = cost
            xbest_brute = x
    return best_cost_brute, xbest_brute


def reverse_array_index_bit_order(arr):
    arr = np.array(arr)
    n = int(np.log2(len(arr)))  # Calculate the value of N
    if n % 1:
        raise ValueError("Input vector has to have length 2**N where N is integer")

    index_arr = np.arange(len(arr))
    new_index_arr = np.zeros_like(index_arr)
    while n > 0:
        last_8 = np.unpackbits(index_arr.astype(np.uint8), axis=0, bitorder="little")
        repacked_first_8 = np.packbits(last_8).astype(np.int64)
        if n < 8:
            new_index_arr += repacked_first_8 >> (8 - n)
        else:
            new_index_arr += repacked_first_8 << (n - 8)
        index_arr = index_arr >> 8
        n -= 8
    return arr[new_index_arr]


def state_to_ampl_counts(vec, eps: float = 1e-15):
    """Converts a statevector to a dictionary
    of bitstrings and corresponding amplitudes
    """
    qubit_dims = np.log2(vec.shape[0])
    if qubit_dims % 1:
        raise ValueError("Input vector is not a valid statevector for qubits.")
    qubit_dims = int(qubit_dims)
    counts = {}
    str_format = "0{}b".format(qubit_dims)
    for kk in range(vec.shape[0]):
        val = vec[kk]
        if val.real**2 + val.imag**2 > eps:
            counts[format(kk, str_format)] = val
    return counts


def precompute_energies(obj_f, nbits: int, *args: object, **kwargs: object):
    """
    Precomputed a vector of objective function values
    that accelerates the energy computation in obj_from_statevector

    For LABS-specific, accelerated version see get_precomputed_labs_merit_factors in qaoa_objective_labs.py


    Parameters
    ----------
    obj_f : callable
        Objective function to precompute
    nbits : int
        Number of parameters obj_f takes
    num_processes : int
        Number of processes to use. Default: 1 (serial)
        if num_processes > 1, pathos.Pool is used
    *args, **kwargs : Objec
        Parameters to be passed directly to obj_f

    Returns
    -------
    energies : np.array
        vector of energies such that E = energies.dot(amplitudes)
        where amplitudes are absolute values squared of qiskit statevector

    """
    bit_strings = (((np.array(range(2**nbits))[:, None] & (1 << np.arange(nbits)))) > 0).astype(int)

    return np.array([obj_f(x, *args, **kwargs) for x in bit_strings])


def yield_all_bitstrings(nbits: int):
    """
    Helper function to avoid having to store all bitstrings in memory
    nbits : int
        Number of parameters obj_f takes
    """
    for x in product([0, 1], repeat=nbits):
        yield np.array(x[::-1])


def precompute_energies_parallel(obj_f, nbits: int, num_processes: int, postfix: list = []):
    """
    Precomputed a vector of objective function values
    that accelerates the energy computation in obj_from_statevector
    Uses pathos.Pool

    For LABS-specific, accelerated version see get_precomputed_labs_merit_factors in qaoa_objective_labs.py


    Parameters
    ----------
    obj_f : callable
        Objective function to precompute
    nbits : int
        Number of parameters obj_f takes
    num_processes : int
        Number of processes to use in pathos.Pool
    postfix : list
        the last k bits

    Returns
    -------
    energies : np.array
        vector of energies such that E = energies.dot(amplitudes)
        where amplitudes are absolute values squared of qiskit statevector

    """
    bit_strings = (np.hstack([x, postfix]) for x in yield_all_bitstrings(nbits - len(postfix)))

    with Pool(num_processes) as pool:
        ens = np.array(pool.map(obj_f, bit_strings))
    return ens


def obj_from_statevector(sv, obj_f, precomputed_energies=None):
    """Compute objective from Qiskit statevector
    For large number of qubits, this is slow.
    """
    if precomputed_energies is None:
        qubit_dims = np.log2(sv.shape[0])
        if qubit_dims % 1:
            raise ValueError("Input vector is not a valid statevector for qubits.")
        qubit_dims = int(qubit_dims)
        # get bit strings for each element of the state vector
        # https://stackoverflow.com/questions/22227595/convert-integer-to-binary-array-with-suitable-padding
        bit_strings = (((np.array(range(sv.shape[0]))[:, None] & (1 << np.arange(qubit_dims)))) > 0).astype(int)

        return sum(obj_f(bit_strings[kk]) * (np.abs(sv[kk]) ** 2) for kk in range(sv.shape[0]))
    else:
        probabilities = np.abs(sv) ** 2
        return precomputed_energies.dot(probabilities)


def unitary_from_circuit(qc: QuantumCircuit):
    backend = qiskit.BasicAer.get_backend("unitary_simulator")
    job = qiskit.execute(qc, backend)
    U = job.result().get_unitary()
    return U


def get_ramp(delta, p: int):
    gamma = np.array([-delta * j / (p + 1) for j in range(1, p + 1)])
    beta = np.array([delta * (1 - j / (p + 1)) for j in range(1, p + 1)])
    return {"beta": beta, "gamma": gamma}


def transpile_hseries(quantinuum_backend: QuantinuumBackend, circuit: Circuit, num_passes_repeats: int = 2):
    """
    Transpile circuit to quantinuum backend
    circuit is qiskit.QuantumCircuit or pytket.circuit.Circuit
    """
    assert isinstance(quantinuum_backend, QuantinuumBackend)

    if isinstance(circuit, QuantumCircuit):
        circ = qiskit_to_tk(circuit)
    else:
        assert isinstance(circuit, Circuit)
        circ = circuit

    compiled_circuit = circ.copy()

    squash = auto_squash_pass({OpType.PhasedX, OpType.Rz})

    fidelities = {
        "ZZMax_fidelity": 0.999,
        "ZZPhase_fidelity": lambda x: 1.0 if not np.isclose(x, 0.5) else 0.9,
    }
    _xcirc = Circuit(1).add_gate(OpType.PhasedX, [1, 0], [0])
    _xcirc.add_phase(0.5)
    passes = [
        RemoveRedundancies(),
        CommuteThroughMultis(),
        FullPeepholeOptimise(target_2qb_gate=OpType.TK2),
        NormaliseTK2(),
        DecomposeTK2(**fidelities),
        quantinuum_backend.rebase_pass(),
        squash,
        SimplifyInitial(allow_classical=False, create_all_qubits=True, xcirc=_xcirc),
    ]

    seqpass = SequencePass(passes)

    # repeat compilation steps `num_passes_repeats` times
    for _ in range(num_passes_repeats):
        seqpass.apply(compiled_circuit)

    ZZMax_depth = compiled_circuit.depth_by_type(OpType.ZZMax)
    ZZPhase_depth = compiled_circuit.depth_by_type(OpType.ZZPhase)
    ZZMax_count = len(compiled_circuit.ops_of_type(OpType.ZZMax))
    ZZPhase_count = len(compiled_circuit.ops_of_type(OpType.ZZPhase))
    two_q_depth = ZZMax_depth + ZZPhase_depth
    two_q_count = ZZMax_count + ZZPhase_count

    return compiled_circuit, {
        "two_q_count": two_q_count,
        "two_q_depth": two_q_depth,
        "ZZMAX": ZZMax_count,
        "ZZPhase": ZZPhase_count,
    }


def objective_from_counts(counts, obj):
    """Compute expected value of the objective from shot counts"""
    mean = 0
    for meas, meas_count in counts.items():
        obj_for_meas = obj(np.array([int(x) for x in meas]))
        mean += obj_for_meas * meas_count
    return mean / sum(counts.values())


def invert_counts(counts):
    """Convert from lsb to msb ordering and vice versa"""
    return {k[::-1]: v for k, v in counts.items()}
