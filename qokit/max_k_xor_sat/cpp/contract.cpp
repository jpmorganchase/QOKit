/**
 * Top-level orchestration: branch levels 1..p, root contraction.
 * Templated on Scalar (C128 or DDComplex).
 */

#include "contract.h"
#include "adjoint.h"
#include "branch.h"
#include "root.h"
#include "grad.h"
#include "dispatch.h"
#include <cmath>
#include <cstdio>
#include <type_traits>
#include <vector>


long long light_cone_size(int p, int D, int k) {
    long long a = (long long)(D - 1) * (k - 1);
    if (a <= 0) return k;
    if (a == 1) return (long long)k * (p + 1);
    constexpr long long LLONG_LIMIT = 4611686018427387903LL;
    long long pow_a = 1;
    for (int i = 0; i <= p; i++) {
        if (pow_a > LLONG_LIMIT / a) return -1;
        pow_a *= a;
    }
    return (long long)k * (pow_a - 1) / (a - 1);
}


// Branch and root dispatch
static std::vector<C128> branch(const double* g, const double* b,
                                 int nr, int k, const C128* child, bool v) {
    return hyperedge_branch(g, b, nr, k, child, v);
}
static std::vector<DDComplex> branch(const double* g, const double* b,
                                      int nr, int k, const DDComplex* child, bool v) {
    return hyperedge_branch_dd(g, b, nr, k, child, v);
}
static double root(const C128* rb, const double* g, const double* b,
                    int p, int D, int k, bool v, bool mutable_rb = false) {
    return root_contract(rb, g, b, p, D, k, v, mutable_rb);
}
static double root(const DDComplex* rb, const double* g, const double* b,
                    int p, int D, int k, bool v, bool mutable_rb = false) {
    return root_contract_dd(rb, g, b, p, D, k, v, mutable_rb);
}


template<typename Scalar>
static double contract_impl(const double* gammas, const double* betas,
                            int p, int D, int k, bool verbose) {
    double log_scale = 0.0;

    if (verbose) fprintf(stderr, "  contract: level 1/%d (leaf)\n", p);
    std::vector<Scalar> F = branch(gammas, betas, 1, k,
                                    (const Scalar*)nullptr, verbose);

    for (int level = 1; level < p; level++) {
        size_t n = F.size();
        size_t mem_mb = n * sizeof(Scalar) / (1024*1024);
        if (verbose)
            fprintf(stderr, "  contract: level %d/%d (4^%d = %zu entries, %zu MB)\n",
                    level + 1, p, level + 1, n * 4, mem_mb * 4);
        std::vector<Scalar> child(n);
        hint_alloc(child.data(), n * sizeof(Scalar));
        // use_dd_power for C128: compute pow in DD precision to reduce noise
        constexpr bool dd_pow = std::is_same<Scalar, C128>::value;
        double F_max = normalize_and_pow(F.data(), child.data(), n, D - 1, dd_pow);
        if (F_max > 0) log_scale += (D - 1) * std::log(F_max);
        { std::vector<Scalar>().swap(F); }  // free F before branch allocates
        F = branch(gammas, betas, level + 1, k, child.data(), verbose);
    }

    if (verbose) fprintf(stderr, "  contract: normalize + power for root\n");
    constexpr bool dd_pow = std::is_same<Scalar, C128>::value;
    double F_max = normalize_and_pow(F.data(), F.data(), F.size(), D - 1, dd_pow);
    if (F_max > 0) log_scale += (D - 1) * std::log(F_max);

    if (verbose) fprintf(stderr, "  contract: root contraction\n");
    double raw = root(F.data(), gammas, betas, p, D, k, verbose, /*mutable_rb=*/true);
    if (log_scale != 0.0) raw *= std::exp(k * log_scale);
    if (verbose) fprintf(stderr, "  contract: done\n");
    return raw;
}


double contract_symmetric_tree(const double* gammas, const double* betas,
                               int p, int D, int k, bool use_dd, bool verbose) {
    if (p == 0) return 0.0;
    return use_dd ? contract_impl<DDComplex>(gammas, betas, p, D, k, verbose)
                  : contract_impl<C128>(gammas, betas, p, D, k, verbose);
}


double contract_with_grad(const double* gammas, const double* betas,
                          int p, int D, int k,
                          double* grad_gammas, double* grad_betas,
                          bool use_dd) {
    return use_dd
        ? gradient_adjoint<DDComplex>(gammas, betas, p, D, k, grad_gammas, grad_betas)
        : gradient_adjoint<C128>(gammas, betas, p, D, k, grad_gammas, grad_betas);
}
