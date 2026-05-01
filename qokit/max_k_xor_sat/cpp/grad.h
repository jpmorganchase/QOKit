#pragma once
/**
 * Gradient via central finite differences, parallelized across parameters.
 *
 * On a 91-core CPU, all 4p perturbations (p gammas forward/backward +
 * p betas forward/backward) run in parallel, each computing a full
 * contraction independently. This turns O(4p) sequential contractions
 * into O(1) wall-clock contractions (given enough cores).
 */

#include <complex>

using C128 = std::complex<double>;

/**
 * Compute <Z^{otimes k}> and its gradient via central finite differences.
 *
 * All 4p perturbation evaluations run in parallel using OpenMP.
 * Each thread gets its own copy of the angles and runs an independent
 * contraction — no shared mutable state.
 *
 * @param gammas      Phase separator angles, length p.
 * @param betas       Mixer angles, length p.
 * @param p           QAOA depth.
 * @param D           Vertex degree.
 * @param k           Hyperedge size.
 * @param grad_gammas Output: gradient w.r.t. gammas, length p.
 * @param grad_betas  Output: gradient w.r.t. betas, length p.
 * @param h           Step size for finite differences (default: 1e-7).
 * @return The expectation value <Z^{otimes k}>.
 */
/**
 * @param h  Step size for finite differences.  If h <= 0, an adaptive
 *           step is computed based on the estimated noise floor:
 *           h = cbrt(sqrt(a)^p * eps), where a=(D-1)(k-1) and eps is
 *           machine epsilon for the chosen precision.
 */
double gradient_fd(const double* gammas, const double* betas,
                   int p, int D, int k,
                   double* grad_gammas, double* grad_betas,
                   double h = 0.0, bool use_dd = false);

/**
 * Compute value + exact gradient via reverse-mode adjoint.
 *
 * Cost: ~3x one forward evaluation (vs 4p+1 for FD).
 * Memory: ~1.67x 4^p for cached intermediates.
 * Float64 only (no DD support yet).
 */
double gradient_adjoint(const double* gammas, const double* betas,
                         int p, int D, int k,
                         double* grad_gammas, double* grad_betas);
