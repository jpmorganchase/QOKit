/**
 * Fused DD kernels with inlined Dekker/Knuth arithmetic.
 *
 * Compiled with -ffp-contract=off -mno-fma (qaoa_dd CMake target).
 * Inlining dd_cmul/dd_cpow here avoids cross-TU function call overhead
 * that the compiler cannot optimize through.
 */

#include "dd_fused.h"
#include "alloc.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif


// ── Inlined DD complex multiply ───────────────────────────────
// Same algorithm as dd_cmul() in dd.cpp, but static inline here
// so the compiler can schedule instructions within the hot loops.

static inline DDComplex dd_cmul_inline(DDComplex a, DDComplex b) {
    double ar = a.hi.real(), ai = a.hi.imag();
    double br = b.hi.real(), bi = b.hi.imag();

    // Real part: ar*br - ai*bi
    double p1, e1, p2, e2;
    two_prod(ar, br, p1, e1);
    two_prod(ai, bi, p2, e2);
    double real_hi = p1 - p2;
    double real_lo = e1 - e2 + ((p1 - real_hi) - p2);

    // Imaginary part: ar*bi + ai*br
    double p3, e3, p4, e4;
    two_prod(ar, bi, p3, e3);
    two_prod(ai, br, p4, e4);
    double imag_hi = p3 + p4;
    double imag_lo = e3 + e4 + ((p3 - imag_hi) + p4);

    C128 hi(real_hi, imag_hi);
    C128 lo(real_lo, imag_lo);

    // Cross terms (lower-order corrections)
    lo += a.hi * b.lo + a.lo * b.hi;

    // Renormalize
    double sr, er, sii, eii;
    two_sum(hi.real(), lo.real(), sr, er);
    two_sum(hi.imag(), lo.imag(), sii, eii);
    return {C128(sr, sii), C128(er, eii)};
}


// ── Inlined DD complex power ──────────────────────────────────

static inline DDComplex dd_cpow_inline(DDComplex base, int n) {
    if (n == 0) return {C128(1.0), C128(0.0)};
    if (n == 1) return base;
    DD_POW_SPECIALIZATIONS(base, dd_cmul_inline);

    DDComplex result = {C128(1.0), C128(0.0)};
    DDComplex b = base;
    int exp = n;
    while (exp > 0) {
        if (exp & 1) result = dd_cmul_inline(result, b);
        b = dd_cmul_inline(b, b);
        exp >>= 1;
    }
    return result;
}


// ── Fused WHT charge contraction ──────────────────────────────

void wht_charge_contract_dd_fused(const DDComplex* M, const DDComplex* T,
                                   DDComplex* out, size_t R, size_t rest) {
    size_t vec_len = 4 * rest;

    #pragma omp parallel for schedule(static) collapse(3)
    for (size_t i = 0; i < R; i++) {
        for (size_t b = 0; b < 4; b++) {
            for (size_t r = 0; r < rest; r++) {
                size_t t_base = i * 16 * rest + b * rest + r;
                size_t stride = 4 * rest;

                DDComplex e0 = dd_cmul_inline(M[b * 4 + 0], T[t_base + 0 * stride]);
                DDComplex e1 = dd_cmul_inline(M[b * 4 + 1], T[t_base + 1 * stride]);
                DDComplex e2 = dd_cmul_inline(M[b * 4 + 2], T[t_base + 2 * stride]);
                DDComplex e3 = dd_cmul_inline(M[b * 4 + 3], T[t_base + 3 * stride]);

                // WHT butterfly: additions are already inline via dd.h
                DDComplex p02 = dd_cadd(e0, e2), q02 = dd_csub(e0, e2);
                DDComplex p13 = dd_cadd(e1, e3), q13 = dd_csub(e1, e3);

                size_t base_out = i * vec_len + b * rest + r;
                out[0 * R * vec_len + base_out] = dd_cadd(p02, p13);
                out[1 * R * vec_len + base_out] = dd_csub(p02, p13);
                out[2 * R * vec_len + base_out] = dd_cadd(q02, q13);
                out[3 * R * vec_len + base_out] = dd_csub(q02, q13);
            }
        }
    }
}


// ── Fused DD mode products ────────────────────────────────────

void mode_products_dd_fused(DDComplex* F, size_t total, int num_rounds,
                            const DDComplex (*W)[16], DDComplex* workspace) {
    DDComplex* src = F;
    DDComplex* dst = workspace;

    // Precompute suffix for each round (all powers of 4)
    size_t suffix_arr[32];
    suffix_arr[num_rounds - 1] = 1;
    for (int i = num_rounds - 2; i >= 0; i--) suffix_arr[i] = suffix_arr[i + 1] * 4;

    for (int ell = 0; ell < num_rounds; ell++) {
        size_t suffix = suffix_arr[ell];
        size_t stride = 4 * suffix;

        #pragma omp parallel for schedule(static)
        for (size_t flat = 0; flat < total; flat++) {
            size_t p_idx = flat / stride;
            size_t rem = flat % stride;
            int h = (int)(rem / suffix);
            size_t s = rem % suffix;

            DDComplex sum;
            size_t base = p_idx * stride + s;
            for (int j = 0; j < 4; j++) {
                sum = dd_cadd(sum, dd_cmul_inline(W[ell][h * 4 + j],
                              src[base + j * suffix]));
            }
            dst[p_idx * stride + h * suffix + s] = sum;
        }

        std::swap(src, dst);
    }

    if (src != F) {
        std::copy(src, src + total, F);
    }
}


// ── Fused normalize + power ───────────────────────────────────

double normalize_and_pow_dd(const DDComplex* src, DDComplex* dst,
                            size_t n, int exponent) {
    // Pass 1: max_abs reduction
    double max_val = 0.0;
    #pragma omp parallel for reduction(max:max_val) schedule(static)
    for (size_t i = 0; i < n; i++) {
        double mag = std::abs(src[i].to_c128());
        if (mag > max_val) max_val = mag;
    }

    if (max_val == 0.0) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) {
            dst[i] = DDComplex();
        }
        return 0.0;
    }

    // Pass 2: fused normalize + power
    // Compute inverse in DD to avoid injecting float64 truncation error
    // into every element before power amplification.
    DD inv_dd = dd_div(DD(1.0), DD(max_val));
    DDComplex inv(C128(inv_dd.hi), C128(inv_dd.lo));
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) {
        dst[i] = dd_cpow_inline(dd_cmul_inline(src[i], inv), exponent);
    }

    return max_val;
}


// ── NUMA first-touch zero ─────────────────────────────────────

void numa_first_touch_zero_dd(DDComplex* data, size_t n) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) {
        data[i] = DDComplex();
    }
}
