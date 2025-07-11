# Changelog

All notable changes to **QOKit (portfolio fork)** will be documented here.

## [0.3.0] – 2025-07-11
### Added
- Benchmark harness `benchmark_vs_bruteforce.py` comparing enhanced
  pipeline to pure-Python brute force for N = 16–25, p = 1–15.

- **Batch-aware portfolio objective**  
  `get_qaoa_portfolio_objective` now accepts a stacked θ array
  `(B, 2 p)` or `(B, p)` and returns a length-B energy vector.  
  *Speed-up*: 2–3 × for sweeps/optimisers.

- **Analytic-gradient path + SciPy L-BFGS-B**  
  `scripts/run_sweep.py --optim lbfgs` uses the analytic Jacobian exposed
  by QOKit, cutting objective evaluations by ~50 – 75 %.

- **Diagonal phase-vector cache** in `apply_qaoa_furxy_ring`  
  Memoises `exp(-0.5 i γ H_diag)` keyed on `(γ, id(H_diag))`.  
  *Speed-up*: 10–25 % on large p sweeps.

### Changed
- **Vectorised cost evaluator** (`get_configuration_cost_vector`) now
  handles both `(N,)` and `(B, N)` in one call (NumPy / CuPy).

### Fixed
- State-vector reuse bug in batched objective (restores pristine |ψ₀⟩
  before every evaluation).

### Benchmark (12 assets, K = 4, q = 0.7, p = 1-3, 8-core CPU)

| Version | Runtime | Objective calls |
|---------|---------|-----------------|
| 0.2.0   | 63 s    | 1 020           |
| 0.3.0   | **23 s**| **320**         |

(Tests executed with `python scripts/run_sweep.py --optim lbfgs`)

