#pragma once
/**
 * Root contraction with factored rank-1 representation.
 */

#include <complex>
#include <vector>
#include "dd.h"

using C128 = std::complex<double>;

/**
 * Root contraction for float64.
 *
 * @param rb       Branch tensor raised to (D-1) power, 4^p entries (flat).
 * @param gammas   Phase angles, length p.
 * @param betas    Mixer angles, length p.
 * @param p        QAOA depth.
 * @param D        Vertex degree.
 * @param k        Hyperedge size.
 * @param verbose  Print timing to stderr.
 * @return <Z^{otimes k}> (real scalar).
 */
double root_contract(const C128* rb, const double* gammas, const double* betas,
                     int p, int D, int k, bool verbose = false,
                     bool mutable_rb = false);

/**
 * Root contraction for DD precision.
 */
double root_contract_dd(const DDComplex* rb, const double* gammas, const double* betas,
                        int p, int D, int k, bool verbose = false,
                        bool mutable_rb = false);
