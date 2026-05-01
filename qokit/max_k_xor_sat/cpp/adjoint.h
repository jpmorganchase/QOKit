#pragma once
/**
 * Reverse-mode adjoint gradient for QAOA symmetric tree contraction.
 *
 * Computes exact gradient in ~3x one forward eval, replacing O(4p)
 * finite-difference evaluations with a single forward+backward pass.
 *
 * Templated on Scalar (C128 or DDComplex) for float64 and DD precision.
 */

#include <complex>
#include <vector>
#include "dd.h"

using C128 = std::complex<double>;

/**
 * Cached forward-pass state for the backward sweep.
 */
template<typename Scalar>
struct ForwardCache {
    int p, D, k;
    double log_scale;
    double raw_result;

    std::vector<double> F_max;                   // normalization max per level
    std::vector<std::vector<Scalar>> F_norm;     // normalized F before power
    std::vector<Scalar> rb;                      // root branch tensor
    double rb_F_max;
};

template<typename Scalar>
ForwardCache<Scalar> forward_pass_cached(const double* gammas, const double* betas,
                                          int p, int D, int k);

template<typename Scalar>
std::vector<Scalar> backward_branch(const Scalar* adj_F,
                                     const double* gammas, const double* betas,
                                     int num_rounds, int k,
                                     const Scalar* child_branch,
                                     double* grad_gammas, double* grad_betas);

template<typename Scalar>
std::vector<Scalar> backward_root(double adj_raw,
                                   const Scalar* rb,
                                   const double* gammas, const double* betas,
                                   int p, int D, int k,
                                   double* grad_gammas, double* grad_betas);

template<typename Scalar>
double gradient_adjoint(const double* gammas, const double* betas,
                         int p, int D, int k,
                         double* grad_gammas, double* grad_betas);


// ── Adjoint primitives ──────────────────────────────────────────

template<typename Scalar>
void wht_charge_contract_adjoint(const Scalar* M, const Scalar* T,
                                  const Scalar* adj_out,
                                  Scalar* adj_T, Scalar* adj_M,
                                  size_t R, size_t rest);

template<typename Scalar>
void mode_product_adjoint(const Scalar* W, const Scalar* src,
                           const Scalar* adj_dst,
                           Scalar* adj_src, Scalar* adj_W,
                           size_t total, int ell, int num_rounds);
