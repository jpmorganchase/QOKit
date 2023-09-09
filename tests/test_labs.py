###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import numpy as np
from qokit.utils import brute_force

from qokit.labs import (
    true_optimal_mf,
    merit_factor,
    slow_merit_factor,
    get_term_indices,
    get_depth_optimized_terms,
    get_gate_optimized_terms_naive,
    get_gate_optimized_terms_greedy,
    true_optimal_energy,
    energy_vals,
    get_energy_term_indices,
    energy_vals_general,
)


def test_slow_merit_factor():
    for N in range(3, 12):
        terms, offset = get_energy_term_indices(N)
        s = np.random.choice([-1, 1], size=N)
        assert np.isclose(
            slow_merit_factor(s, terms=terms, offset=offset, check_parameters=True),
            merit_factor(s, N),
        )


def test_merit_factor():
    for N in range(3, 12):
        assert np.around(brute_force(merit_factor, N)[0], decimals=3) == true_optimal_mf[N]


def test_depth_opt_terms_generation():
    for N in range(3, 16):
        assert set(get_term_indices(N)) == set(get_depth_optimized_terms(N))


def test_gate_optimized_terms_naive():
    for N in range(3, 16):
        assert set(get_term_indices(N)) == set(get_gate_optimized_terms_naive(N, number_of_gate_zones=4))
        assert set(get_term_indices(N)) == set(get_gate_optimized_terms_naive(N, number_of_gate_zones=None))


def test_gate_optimized_terms_greedy():
    for N in range(3, 16):
        assert set(get_term_indices(N)) == set(get_gate_optimized_terms_greedy(N, number_of_gate_zones=4))
        assert set(get_term_indices(N)) == set(get_gate_optimized_terms_greedy(N, number_of_gate_zones=None))


def test_energy_vals():
    for N in range(3, 12):
        assert brute_force(energy_vals, N, minimize=True)[0] == true_optimal_energy[N]
    for N in range(
        max(min(true_optimal_energy.keys()), min(true_optimal_mf.keys())),
        min(max(true_optimal_energy.keys()), max(true_optimal_mf.keys())),
    ):
        mf_from_en = N**2 / (2 * true_optimal_energy[N])
        assert np.around(mf_from_en, decimals=3) == true_optimal_mf[N]
    for N in range(3, 12):
        terms, offset = get_energy_term_indices(N)
        s = np.random.choice([-1, 1], size=N)
        assert np.isclose(
            energy_vals_general(s, terms=terms, offset=offset, check_parameters=True),
            energy_vals(s, N=N),
        )
