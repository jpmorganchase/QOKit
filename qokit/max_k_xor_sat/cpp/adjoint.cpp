/**
 * Top-level adjoint gradient: cached forward pass + backward sweep.
 * Templated on Scalar (C128 or DDComplex).
 *
 * Cost: ~3x one forward evaluation.
 * Memory: ~1.67x 4^p for cached intermediates.
 */

#include "adjoint.h"
#include "branch.h"
#include "root.h"
#include "primitives.h"
#include "dispatch.h"
#include <cmath>
#include <cstring>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif


// Branch/root dispatch (same pattern as contract.cpp)
static std::vector<C128> branch_fwd(const double* g, const double* b,
                                     int nr, int k, const C128* child) {
    return hyperedge_branch(g, b, nr, k, child);
}
static std::vector<DDComplex> branch_fwd(const double* g, const double* b,
                                          int nr, int k, const DDComplex* child) {
    return hyperedge_branch_dd(g, b, nr, k, child);
}
static double root_fwd(const C128* rb, const double* g, const double* b,
                        int p, int D, int k) {
    return root_contract(rb, g, b, p, D, k);
}
static double root_fwd(const DDComplex* rb, const double* g, const double* b,
                        int p, int D, int k) {
    return root_contract_dd(rb, g, b, p, D, k);
}


template<typename Scalar>
ForwardCache<Scalar> forward_pass_cached(const double* gammas, const double* betas,
                                          int p, int D, int k) {
    ForwardCache<Scalar> cache;
    cache.p = p;
    cache.D = D;
    cache.k = k;
    cache.log_scale = 0.0;
    cache.F_max.resize(p, 0.0);
    cache.F_norm.resize(p);

    // Level 1: leaf
    std::vector<Scalar> F = branch_fwd(gammas, betas, 1, k, (const Scalar*)nullptr);

    for (int level = 1; level < p; level++) {
        // Normalize before (D-1) power
        double F_max_val = 0.0;
        #pragma omp parallel for reduction(max:F_max_val) schedule(static)
        for (size_t i = 0; i < F.size(); i++) {
            double v = abs_scalar(F[i]);
            if (v > F_max_val) F_max_val = v;
        }
        cache.F_max[level - 1] = F_max_val;

        if (F_max_val > 0) {
            double inv = 1.0 / F_max_val;
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < F.size(); i++)
                F[i] = F[i] * inv;
            cache.log_scale += (D - 1) * std::log(F_max_val);
        }
        cache.F_norm[level - 1] = F;

        size_t child_size = F.size();
        std::vector<Scalar> child(child_size);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < child_size; i++)
            child[i] = pow_scalar(F[i], D - 1);

        F = branch_fwd(gammas, betas, level + 1, k, child.data());
    }

    // Final normalization before root (D-1) power
    double F_max_val = 0.0;
    #pragma omp parallel for reduction(max:F_max_val) schedule(static)
    for (size_t i = 0; i < F.size(); i++) {
        double v = abs_scalar(F[i]);
        if (v > F_max_val) F_max_val = v;
    }
    cache.F_max[p - 1] = F_max_val;
    cache.rb_F_max = F_max_val;

    if (F_max_val > 0) {
        double inv = 1.0 / F_max_val;
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < F.size(); i++)
            F[i] = F[i] * inv;
        cache.log_scale += (D - 1) * std::log(F_max_val);
    }
    cache.F_norm[p - 1] = F;

    std::vector<Scalar> rb(F.size());
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < F.size(); i++)
        rb[i] = pow_scalar(F[i], D - 1);
    cache.rb = rb;

    cache.raw_result = root_fwd(rb.data(), gammas, betas, p, D, k);
    return cache;
}


template<typename Scalar>
double gradient_adjoint(const double* gammas, const double* betas,
                         int p, int D, int k,
                         double* grad_gammas, double* grad_betas) {
    if (p == 0) return 0.0;

    std::memset(grad_gammas, 0, p * sizeof(double));
    std::memset(grad_betas, 0, p * sizeof(double));

    auto cache = forward_pass_cached<Scalar>(gammas, betas, p, D, k);

    double result = cache.raw_result;
    if (cache.log_scale != 0.0)
        result *= std::exp(k * cache.log_scale);

    double scale_factor = (cache.log_scale != 0.0)
                          ? std::exp(k * cache.log_scale) : 1.0;
    double adj_raw = scale_factor;

    // Backward through root contraction
    std::vector<Scalar> adj_rb = backward_root<Scalar>(
        adj_raw, cache.rb.data(), gammas, betas, p, D, k,
        grad_gammas, grad_betas);

    // Backward through final normalization + (D-1) power
    size_t F_size = cache.F_norm[p - 1].size();
    std::vector<Scalar> adj_F_norm(F_size, Scalar(C128(0.0)));

    if (D - 1 == 1) {
        adj_F_norm = adj_rb;
    } else {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < F_size; i++) {
            Scalar fn = cache.F_norm[p - 1][i];
            if (abs_scalar(fn) > 0) {
                Scalar deriv = Scalar((double)(D - 1)) * pow_scalar(fn, D - 2);
                adj_F_norm[i] = adj_rb[i] * conj_scalar(deriv);
            }
        }
    }

    double F_max_val = cache.rb_F_max;
    std::vector<Scalar> adj_F(F_size);
    if (F_max_val > 0) {
        double inv_fmax = 1.0 / F_max_val;
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < F_size; i++)
            adj_F[i] = adj_F_norm[i] * inv_fmax;
    } else {
        adj_F = adj_F_norm;
    }

    // Backward through levels p down to 1
    for (int level = p - 1; level >= 1; level--) {
        size_t child_size = cache.F_norm[level - 1].size();
        std::vector<Scalar> child(child_size);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < child_size; i++)
            child[i] = pow_scalar(cache.F_norm[level - 1][i], D - 1);

        std::vector<Scalar> adj_child = backward_branch<Scalar>(
            adj_F.data(), gammas, betas, level + 1, k,
            child.data(), grad_gammas, grad_betas);

        std::vector<Scalar> adj_fn(child_size, Scalar(C128(0.0)));
        if (D - 1 == 1) {
            adj_fn = adj_child;
        } else {
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < child_size; i++) {
                Scalar fn = cache.F_norm[level - 1][i];
                if (abs_scalar(fn) > 0) {
                    Scalar deriv = Scalar((double)(D - 1)) * pow_scalar(fn, D - 2);
                    adj_fn[i] = adj_child[i] * conj_scalar(deriv);
                }
            }
        }

        double fmax = cache.F_max[level - 1];
        adj_F.resize(child_size);
        if (fmax > 0) {
            double inv_fm = 1.0 / fmax;
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < child_size; i++)
                adj_F[i] = adj_fn[i] * inv_fm;
        } else {
            adj_F = adj_fn;
        }
    }

    // Leaf backward
    backward_branch<Scalar>(adj_F.data(), gammas, betas, 1, k,
                             (const Scalar*)nullptr, grad_gammas, grad_betas);

    return result;
}


// Explicit instantiations
template ForwardCache<C128> forward_pass_cached<C128>(const double*, const double*, int, int, int);
template ForwardCache<DDComplex> forward_pass_cached<DDComplex>(const double*, const double*, int, int, int);
template double gradient_adjoint<C128>(const double*, const double*, int, int, int, double*, double*);
template double gradient_adjoint<DDComplex>(const double*, const double*, int, int, int, double*, double*);
