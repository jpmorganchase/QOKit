#pragma once
/**
 * Branch contraction: Phase 1, Phase 2, mode products.
 */

#include <complex>
#include <vector>
#include "dd.h"

using C128 = std::complex<double>;

/**
 * Branch tensor for one hyperedge (4^num_rounds entries).
 *
 * @param gammas       Phase angles, length >= num_rounds.
 * @param betas        Mixer angles, length >= num_rounds.
 * @param num_rounds   Number of rounds for this branch.
 * @param k            Hyperedge size.
 * @param child_branch Child branch tensor (4^{num_rounds-1} entries), or nullptr for leaf.
 * @param verbose      Print timing to stderr.
 * @return Flat branch tensor, 4^num_rounds complex entries.
 */
std::vector<C128> hyperedge_branch(const double* gammas, const double* betas,
                                    int num_rounds, int k,
                                    const C128* child_branch,
                                    bool verbose = false);

/// DD version of branch contraction
std::vector<DDComplex> hyperedge_branch_dd(const double* gammas, const double* betas,
                                           int num_rounds, int k,
                                           const DDComplex* child_branch,
                                           bool verbose = false);
