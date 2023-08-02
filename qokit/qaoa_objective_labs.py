###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from __future__ import annotations
from collections.abc import Sequence
import typing
import numpy as np
from pathlib import Path

from .labs import (
    get_energy_term_indices,
    negative_merit_factor_from_bitstring,
    true_optimal_energy,
    energy_vals_from_bitstring,
)

from .utils import precompute_energies
from .qaoa_circuit_labs import get_parameterized_qaoa_circuit
from .qaoa_objective import get_qaoa_objective

qaoa_objective_labs_folder = Path(__file__).parent


class PrecomputedLABSHandler:
    """Singleton handling precomputed LABS merit factors
    and optimal bitstrings, loaded from disk or precomputed
    """

    def __init__(self):
        self.precomputed_merit_factors_dict = {}
        self.precomputed_bitstrings_dict = {}

    def get_precomputed_merit_factors(self, N: int):
        if N in self.precomputed_merit_factors_dict.keys():
            return self.precomputed_merit_factors_dict[N]
        fpath = Path(
            qaoa_objective_labs_folder,
            f"assets/precomputed_merit_factors/precomputed_energies_{N}.npy",
        )
        if N > 10 and fpath.exists():
            # load from disk
            ens = np.load(fpath)
        else:
            # precompute
            if N > 10 and N <= 24:
                raise RuntimeError(
                    f"""
Failed to load from {fpath}, attempting to recompute for N={N},
Precomputed energies should be loaded from disk instead. Run assets/load_assets_from_s3.sh to obtain precomputed energies
                    """
                )
            ens = precompute_energies(negative_merit_factor_from_bitstring, N)
        self.precomputed_merit_factors_dict[N] = ens
        return ens

    def get_precomputed_optimal_bitstrings(self, N: int):
        if N in self.precomputed_bitstrings_dict.keys():
            return self.precomputed_bitstrings_dict[N]
        fpath = Path(
            qaoa_objective_labs_folder,
            f"assets/precomputed_bitstrings/precomputed_bitstrings_{N}.npy",
        )
        if fpath.exists():
            # load from disk
            ens = np.load(fpath)
        else:
            # precompute
            bit_strings = (((np.array(range(2**N))[:, None] & (1 << np.arange(N)))) > 0).astype(int)
            optimal_bitstrings = []
            for x in bit_strings:
                energy = energy_vals_from_bitstring(x, N=N)
                if energy == true_optimal_energy[N]:
                    optimal_bitstrings.append(x)
            ens = optimal_bitstrings
        self.precomputed_bitstrings_dict[N] = ens
        return ens


precomputed_labs_handler = PrecomputedLABSHandler()


def get_precomputed_labs_merit_factors(N: int) -> np.ndarray:
    """
    Return a precomputed a vector of negative LABS merit factors
    that accelerates the energy computation in obj_from_statevector

    If available, loads the precomputed vector from disk

    Parameters
    ----------
    N : int
        Number of spins in the LABS problem

    Returns
    -------
    merit_factors : np.array
        vector of merit factors such that expected merit factor = -energies.dot(probabilities)
        where probabilities are absolute values squared of qiskit statevector
        (minus sign is since the typical use case is optimization)
    """

    return precomputed_labs_handler.get_precomputed_merit_factors(N)


def get_precomputed_optimal_bitstrings(N: int) -> np.ndarray:
    """
    Return a precomputed  optimal bitstring for LABS problem of problem size N

    If available, loads the precomputed vector from disk

    Parameters
    ----------
    N : int
        Number of spins in the LABS problem

    Returns
    -------
    optimal_bitstrings : np.array
    """

    return precomputed_labs_handler.get_precomputed_optimal_bitstrings(N)


def get_random_guess_merit_factor(N: int) -> float:
    """
    Return the merit factor corresponding to the random guess

    Parameters
    ----------
    N : int
        Number of spins in the LABS problem

    Returns
    -------
    MF : float
        Expected merit factor of random guess
    """
    return np.mean(-get_precomputed_labs_merit_factors(N))


def get_qaoa_labs_objective(
    N: int,
    p: int,
    precomputed_merit_factors: np.ndarray | None = None,
    parameterization: str = "theta",
    objective: str = "expectation",
    precomputed_optimal_bitstrings: np.ndarray | None = None,
    simulator: str = "auto",
) -> typing.Callable:
    """Return QAOA objective to be minimized

    Parameters
    ----------
    N : int
        Number of qubits
    p : int
        Number of QAOA layers (number of parameters will be 2*p)
    precomputed_merit_factors : np.array
        precomputed merit factors to compute the QAOA expectation
    parameterization : str
        If parameterization == 'theta', then f takes one parameter (gamma and beta concatenated)
        If parameterization == 'gamma beta', then f takes two parameters (gamma and beta)
        For below Fourier parameters, q=p
        If parameterization == 'freq', then f takes one parameter (fourier parameters u and v concatenated)
        If parameterization == 'u v', then f takes two parameters (fourier parameters u and v)
    objective : str
        If objective == 'expectation', then returns f(theta) = - < theta | C_{LABS} | theta > (minus for minimization)
        If objective == 'overlap', then returns f(theta) = 1 - Overlap |<theta|optimal_bitstring>|^2 (1-overlap for minimization)
        If objective == 'expectation and overlap', then returns a tuple (expectation, overlap)
    precomputed_optimal_bitstrings : np.ndarray
        precomputed optimal bit strings to compute the QAOA overlap
    simulator : str
        If simulator == 'auto', implementation is chosen automatically
            (either the fastest CPU simulator or a GPU simulator if CUDA is available)
        If simulator == 'qiskit', implementation in qaoa_circuit_labs is used

    Returns
    -------
    f : callable
        Function returning the negative of expected value of QAOA with parameters theta
    """

    # TODO: needs to generate parameterized circuit and check that the precomputed stuff is loaded correctly
    # Otherwise pass directly to get_qaoa_objective

    terms_ix, offset = get_energy_term_indices(N)

    if precomputed_merit_factors is None:
        precomputed_merit_factors = get_precomputed_labs_merit_factors(N)

    if objective in ["overlap", "expectation and overlap"] and precomputed_optimal_bitstrings is None:
        precomputed_optimal_bitstrings = get_precomputed_optimal_bitstrings(N)

    precomputed_diagonal_hamiltonian = -(N**2) / (2 * precomputed_merit_factors) - offset

    return get_qaoa_objective(
        N=N,
        p=p,
        precomputed_diagonal_hamiltonian=precomputed_diagonal_hamiltonian,
        precomputed_objectives=precomputed_merit_factors,
        precomputed_optimal_bitstrings=precomputed_optimal_bitstrings,
        parameterization=parameterization,
        objective=objective,
        simulator=simulator,
    )
