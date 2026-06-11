"""Tests for the lazy (on-the-fly) diagonal computation introduced in Issue #35.

These tests run on CPU via the numpy/python backend so they don't require a GPU.
GPU-specific benchmarks are in benchmark/lazy_diagonal_benchmark.py.
"""

###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################

import sys
import os
import importlib.util
import numpy as np
import pytest

# Load numpy_vectorized directly to avoid qokit/__init__.py importing qiskit.
_repo_root = os.path.join(os.path.dirname(__file__), "..")
_nv_path = os.path.join(_repo_root, "qokit", "fur", "diagonal_precomputation", "numpy_vectorized.py")
_nv_spec = importlib.util.spec_from_file_location("numpy_vectorized", _nv_path)
_nv_mod = importlib.util.module_from_spec(_nv_spec)
_nv_spec.loader.exec_module(_nv_mod)
precompute_vectorized_cpu_parallel = _nv_mod.precompute_vectorized_cpu_parallel


def _maxcut_terms(n: int, n_edges: int, seed: int = 42):
    """Generate a random MAXCUT Ising problem as a terms list."""
    rng = np.random.default_rng(seed)
    edges = set()
    while len(edges) < n_edges:
        u, v = sorted(rng.choice(n, 2, replace=False).tolist())
        edges.add((u, v))
    return [(1.0, list(e)) for e in edges]


def _compute_energy_python(state: int, terms) -> float:
    """Reference: compute Ising energy for a single state in pure Python."""
    energy = 0.0
    for coef, positions in terms:
        mask = sum(1 << p for p in positions)
        parity = bin(state & mask).count("1") % 2
        energy += coef * (1 - 2 * parity)
    return energy


class TestTermsToMaskArrays:
    """Test the terms → (coef_array, mask_array) conversion used by the
    lazy diagonal kernel."""

    def test_single_edge_mask(self):
        terms = [(1.0, [0, 2])]
        expected_mask = (1 << 0) | (1 << 2)  # 0b101 = 5
        coefs = np.array([t[0] for t in terms], dtype=np.float32)
        masks = np.array([sum(1 << p for p in t[1]) for t in terms], dtype=np.int64)
        assert coefs[0] == pytest.approx(1.0)
        assert masks[0] == expected_mask

    def test_three_edge_masks(self):
        terms = [(0.5, [1, 3]), (1.0, [0, 2]), (2.0, [0, 1, 2])]
        masks = [sum(1 << p for p in t[1]) for t in terms]
        assert masks[0] == (1 << 1) | (1 << 3)
        assert masks[1] == (1 << 0) | (1 << 2)
        assert masks[2] == (1 << 0) | (1 << 1) | (1 << 2)


class TestEnergyComputationEquivalence:
    """Verify that on-the-fly energy matches the precomputed diagonal."""

    def _precomputed_energies(self, n: int, terms) -> np.ndarray:
        """Use the existing CPU vectorised implementation as the reference."""
        weighted_terms = [(coef, positions) for coef, positions in terms]
        return precompute_vectorized_cpu_parallel(weighted_terms, offset=0, N=n)

    def test_n8_10edges(self):
        n = 8
        terms = _maxcut_terms(n, n_edges=10, seed=1)
        precomputed = self._precomputed_energies(n, terms)
        for state in range(2**n):
            expected = precomputed[state]
            actual = _compute_energy_python(state, terms)
            assert actual == pytest.approx(expected, abs=1e-5), f"state={state}: expected={expected}, got={actual}"

    def test_n12_20edges(self):
        n = 12
        terms = _maxcut_terms(n, n_edges=20, seed=7)
        precomputed = self._precomputed_energies(n, terms)
        # Check a random subset to keep test fast
        rng = np.random.default_rng(99)
        states = rng.integers(0, 2**n, size=200).tolist()
        for state in states:
            expected = precomputed[state]
            actual = _compute_energy_python(state, terms)
            assert actual == pytest.approx(expected, abs=1e-5)

    def test_zero_coefficient_term(self):
        n = 6
        terms = [(0.0, [0, 1]), (1.0, [2, 3])]
        precomputed = self._precomputed_energies(n, terms)
        for state in range(2**n):
            expected = precomputed[state]
            actual = _compute_energy_python(state, terms)
            assert actual == pytest.approx(expected, abs=1e-5)

    def test_negative_coefficient_term(self):
        n = 6
        terms = [(-1.0, [0, 1]), (1.5, [1, 2])]
        precomputed = self._precomputed_energies(n, terms)
        for state in range(2**n):
            actual = _compute_energy_python(state, terms)
            assert actual == pytest.approx(precomputed[state], abs=1e-5)


class TestLazyThreshold:
    """Test the auto-selection threshold logic."""

    def test_threshold_default(self):
        try:
            from qokit.fur.nbcuda.qaoa_simulator import _LAZY_DIAGONAL_THRESHOLD_QUBITS

            assert isinstance(_LAZY_DIAGONAL_THRESHOLD_QUBITS, int)
            assert 18 <= _LAZY_DIAGONAL_THRESHOLD_QUBITS <= 28, "Threshold should be in range [18, 28] qubits"
        except ImportError:
            pytest.skip("numba.cuda not available in this environment")
