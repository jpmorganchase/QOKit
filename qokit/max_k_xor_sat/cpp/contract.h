#pragma once
/**
 * Public API for QAOA symmetric tree contraction.
 */

#include <complex>
#include <vector>

using C128 = std::complex<double>;

/**
 * Number of qubits in the depth-p light cone.
 * N_lc = k * (a^{p+1} - 1) / (a - 1), where a = (D-1)(k-1).
 */
long long light_cone_size(int p, int D, int k);

/**
 * Exact <Z^{otimes k}> for depth-p QAOA on a D-regular k-uniform tree.
 *
 * @param gammas  Phase separator angles, length p.
 * @param betas   Mixer angles, length p.
 * @param p       QAOA depth.
 * @param D       Vertex degree.
 * @param k       Hyperedge size.
 * @param use_dd  If true, use double-double precision.
 * @param verbose If true, print timing to stderr.
 * @return The expectation value <Z^{otimes k}>.
 */
double contract_symmetric_tree(const double* gammas, const double* betas,
                               int p, int D, int k,
                               bool use_dd = false, bool verbose = false);

/**
 * Compute <Z^{otimes k}> and its gradient via forward-mode tangent propagation.
 *
 * @param gammas      Phase separator angles, length p.
 * @param betas       Mixer angles, length p.
 * @param p           QAOA depth.
 * @param D           Vertex degree.
 * @param k           Hyperedge size.
 * @param grad_gammas Output gradient w.r.t. gammas, length p.
 * @param grad_betas  Output gradient w.r.t. betas, length p.
 * @return The expectation value <Z^{otimes k}>.
 */
double contract_with_grad(const double* gammas, const double* betas,
                          int p, int D, int k,
                          double* grad_gammas, double* grad_betas,
                          bool use_dd = false);
