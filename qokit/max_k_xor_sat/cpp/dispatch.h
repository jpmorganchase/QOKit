#pragma once
/**
 * Overloaded dispatch helpers for templated C128/DDComplex code.
 *
 * Included by branch.cpp, root.cpp, contract.cpp to avoid duplicating
 * these overloads in each translation unit.
 */

#include "dd.h"
#include "dd_fused.h"
#include "alloc.h"
#include "primitives.h"
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif

// ── WHT charge contraction ────────────────────────────────────

inline void wht_dispatch(const C128* M, const C128* in, C128* out,
                         size_t R, size_t rest) {
    wht_charge_contract(M, in, out, R, rest);
}
inline void wht_dispatch(const DDComplex* M, const DDComplex* in, DDComplex* out,
                         size_t R, size_t rest) {
    wht_charge_contract_dd_fused(M, in, out, R, rest);
}

// ── Memory hints (no-op for C128) ─────────────────────────────

inline void hint_alloc(C128*, size_t) {}
inline void hint_alloc(DDComplex* p, size_t bytes) { hint_huge_pages(p, bytes); }

inline void first_touch_zero(C128*, size_t) {}
inline void first_touch_zero(DDComplex* p, size_t n) { numa_first_touch_zero_dd(p, n); }

// ── Normalize + power ─────────────────────────────────────────

inline double normalize_and_pow(const C128* src, C128* dst, size_t n, int exponent,
                                bool use_dd_power = false) {
    double max_val = 0.0;
    #pragma omp parallel for reduction(max:max_val) schedule(static)
    for (size_t i = 0; i < n; i++) {
        double mag = std::abs(src[i]);
        if (mag > max_val) max_val = mag;
    }
    if (max_val == 0.0) {
        std::memset(dst, 0, n * sizeof(C128));
        return 0.0;
    }
    double inv = 1.0 / max_val;
    if (use_dd_power) {
        // Hybrid: float64 storage, DD precision for the power step.
        // Eliminates ~4 ULP rounding per pow call that accumulates across levels.
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) {
            DDComplex z(src[i] * inv);  // promote to DD
            dst[i] = dd_cpow(z, exponent).to_c128();  // power in DD, convert back
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) {
            dst[i] = pow_precise(src[i] * inv, exponent);
        }
    }
    return max_val;
}

inline double normalize_and_pow(const DDComplex* src, DDComplex* dst,
                                size_t n, int exponent, bool = false) {
    return normalize_and_pow_dd(src, dst, n, exponent);
}

// ── Precompute modified mixers ────────────────────────────────

template<typename Scalar>
inline void precompute_modified_mixers(const double* betas, int num_rounds,
                                       std::vector<std::array<Scalar, 64>>& md_flat) {
    md_flat.resize(num_rounds);
    for (int ell = 0; ell < num_rounds; ell++) {
        C128 M[16];
        doubled_mixer(betas[ell], M);
        for (int a = 0; a < 4; a++)
            for (int row = 0; row < 4; row++)
                for (int col = 0; col < 4; col++)
                    md_flat[ell][a * 16 + row * 4 + col] =
                        Scalar(M[row * 4 + col] * CHARGE_DIAG[a][col]);
    }
}

// ── Precompute charge weight matrices ─────────────────────────

template<typename Scalar>
inline void precompute_charge_weights(const double* gammas, int num_rounds,
                                      std::vector<std::array<Scalar, 16>>& W) {
    W.resize(num_rounds);
    for (int ell = 0; ell < num_rounds; ell++) {
        C128 W_f64[16];
        charge_weight_matrix(gammas[ell], W_f64);
        for (int i = 0; i < 16; i++) W[ell][i] = Scalar(W_f64[i]);
    }
}

// ── Compute trace matrix from modified mixers ─────────────────

template<typename Scalar>
inline void compute_trace_matrix(const std::vector<std::array<Scalar, 64>>& md_flat,
                                 int num_rounds, Scalar* trace_matrix) {
    for (int a = 0; a < 4; a++)
        for (int s = 0; s < 4; s++)
            trace_matrix[s * 4 + a] =
                md_flat[num_rounds - 1][a * 16 + 0 * 4 + s] +
                md_flat[num_rounds - 1][a * 16 + 3 * 4 + s];
}

// ── Precompute mixer + root charge weights ────────────────────

template<typename Scalar>
inline void precompute_mixer_and_weights(double beta, double gamma,
                                         Scalar* M, Scalar* u) {
    C128 M_f64[16], u_f64[4];
    doubled_mixer(beta, M_f64);
    root_charge_weights(gamma, u_f64);
    for (int i = 0; i < 16; i++) M[i] = Scalar(M_f64[i]);
    for (int i = 0; i < 4; i++) u[i] = Scalar(u_f64[i]);
}
