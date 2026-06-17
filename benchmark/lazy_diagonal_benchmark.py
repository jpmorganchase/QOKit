#!/usr/bin/env python3
"""Benchmark: on-the-fly vs precomputed diagonal for QAOA MAXCUT simulation.

Timing includes diagonal precomputation in the "precomputed" path, so this
represents total wall-clock time per fresh QAOA simulation run.

Measured on RTX 3070Ti Laptop GPU, 100-edge MAXCUT, p=3 QAOA layers:

    n=18  precomputed=12.5ms  lazy= 2.6ms  winner=lazy    (4.9x)
    n=20  precomputed=13.6ms  lazy= 9.3ms  winner=lazy    (1.5x)
    n=22  precomputed=18.8ms  lazy=28.5ms  winner=precomputed
    n=24  precomputed=64.7ms  lazy=114.8ms winner=precomputed
    n=26  precomputed=250.6ms lazy=456.7ms winner=precomputed

Key findings:
- For n <= 20: lazy avoids expensive CPU precomputation + PCIe transfer,
  making it 1.5-5x faster end-to-end.
- For n >= 22 and 100 edges: precomputed is faster per-step, but lazy
  still saves GPU memory (at n=26 the diagonal alone requires 256 MB).
- Primary memory benefit: no 2^n float32 allocation on GPU.
  At n=24: saves 64 MB; at n=26: saves 256 MB; at n=28: saves 1 GB.

Run with:

    python benchmark/lazy_diagonal_benchmark.py
"""
###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################

import time
import numpy as np
import numba.cuda

from qokit.fur.nbcuda.diagonal import (
    apply_diagonal,
    apply_diagonal_from_terms,
    terms_to_device_arrays,
)
from qokit.fur.diagonal_precomputation import precompute_gpu


def make_maxcut_terms(n_qubits: int, n_edges: int, seed: int = 42):
    """Return a random MAXCUT Ising problem as a terms list."""
    rng = np.random.default_rng(seed)
    edges = set()
    while len(edges) < n_edges:
        u, v = sorted(rng.choice(n_qubits, 2, replace=False).tolist())
        edges.add((u, v))
    return [(1.0, list(e)) for e in edges]


def _warmup_gpu():
    a = numba.cuda.device_array(1024, dtype="float32")
    b = numba.cuda.device_array(1024, dtype="complex128")
    numba.cuda.synchronize()


def bench_precomputed(n_qubits: int, terms, gammas, repeats: int = 5):
    n_states = 2 ** n_qubits
    out = numba.cuda.device_array(n_states, dtype="float32")
    precompute_gpu(0, n_qubits, terms, out)
    numba.cuda.synchronize()

    sv = numba.cuda.device_array(n_states, dtype="complex128")
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        # Simulate full QAOA layer: precompute + apply per layer
        diag = numba.cuda.device_array(n_states, dtype="float32")
        precompute_gpu(0, n_qubits, terms, diag)
        for gamma in gammas:
            apply_diagonal(sv, gamma, diag)
        numba.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1000  # ms


def bench_lazy(n_qubits: int, terms, gammas, repeats: int = 5):
    n_states = 2 ** n_qubits
    coef_dev, mask_dev = terms_to_device_arrays(terms)
    numba.cuda.synchronize()

    sv = numba.cuda.device_array(n_states, dtype="complex128")
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for gamma in gammas:
            apply_diagonal_from_terms(sv, gamma, coef_dev, mask_dev)
        numba.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1000  # ms


def verify_correctness(n_qubits: int = 16, n_edges: int = 30, seed: int = 7):
    """Check that lazy and precomputed paths produce identical statevectors."""
    import cmath
    terms = make_maxcut_terms(n_qubits, n_edges, seed=seed)
    n_states = 2 ** n_qubits

    out = numba.cuda.device_array(n_states, dtype="float32")
    precompute_gpu(0, n_qubits, terms, out)
    diag = out.copy_to_host()

    coef_dev, mask_dev = terms_to_device_arrays(terms)

    gamma = 0.5
    sv_pre = numba.cuda.to_device(np.ones(n_states, dtype="complex128") / (n_states ** 0.5))
    sv_lazy = numba.cuda.to_device(np.ones(n_states, dtype="complex128") / (n_states ** 0.5))

    apply_diagonal(sv_pre, gamma, numba.cuda.to_device(diag.astype("float32")))
    apply_diagonal_from_terms(sv_lazy, gamma, coef_dev, mask_dev)
    numba.cuda.synchronize()

    pre = sv_pre.copy_to_host()
    lazy = sv_lazy.copy_to_host()
    max_err = np.max(np.abs(pre - lazy))
    assert max_err < 1e-5, f"Correctness check FAILED: max_err={max_err:.2e}"
    print(f"  Correctness OK (n={n_qubits}, edges={n_edges}, max_err={max_err:.2e})")


def main():
    print("QOKit lazy-diagonal benchmark (Issue #35)")
    print("GPU:", numba.cuda.get_current_device().name.decode())
    print()

    _warmup_gpu()

    print("Correctness check:")
    verify_correctness(16, 30)
    verify_correctness(20, 100)
    print()

    n_edges = 100
    p_layers = 3
    gammas = [0.3, 0.5, 0.7][:p_layers]

    print(f"{'n':>4}  {'precomputed':>14}  {'lazy':>10}  {'speedup':>8}  winner")
    print("-" * 55)

    for n in [18, 20, 22, 24, 26]:
        terms = make_maxcut_terms(n, n_edges)

        t_pre = bench_precomputed(n, terms, gammas)
        t_lazy = bench_lazy(n, terms, gammas)
        speedup = t_pre / t_lazy
        winner = "lazy" if t_lazy < t_pre else "precomputed"

        print(f"{n:>4}  {t_pre:>12.1f}ms  {t_lazy:>8.1f}ms  {speedup:>7.1f}x  {winner}")


if __name__ == "__main__":
    main()
