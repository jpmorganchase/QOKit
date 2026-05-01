#pragma once
/**
 * Fused DD kernels with inlined arithmetic for hot loops.
 *
 * The implementation in dd_fused.cpp is compiled with -ffp-contract=off -mno-fma
 * (same CMake target as dd.cpp) to guarantee Dekker splitting correctness.
 *
 * This header contains only declarations — safe to include from any TU.
 */

#include "dd.h"
#include <cstddef>

/**
 * Fused DD WHT charge contraction with inlined DD multiply.
 * Same interface as wht_charge_contract_dd() in primitives.h.
 */
void wht_charge_contract_dd_fused(const DDComplex* M, const DDComplex* T,
                                   DDComplex* out, size_t R, size_t rest);

/**
 * Full DD mode products with ping-pong buffering and inlined DD multiply.
 *
 * @param F         Input/output tensor, shape (4,)^num_rounds flat
 * @param total     Total elements (4^num_rounds)
 * @param num_rounds Number of axes
 * @param W         Charge weight matrices, num_rounds arrays of 16 DDComplex
 * @param workspace Pre-allocated buffer of size `total` (ping-pong partner)
 */
void mode_products_dd_fused(DDComplex* F, size_t total, int num_rounds,
                            const DDComplex (*W)[16], DDComplex* workspace);

/**
 * Fused normalize + power: two passes instead of three.
 *
 * Pass 1: max_abs reduction over src.
 * Pass 2: dst[i] = dd_cpow(dd_cmul(src[i], 1/max), exponent).
 *
 * @param src       Input array (not modified)
 * @param dst       Output array (must not overlap src)
 * @param n         Number of elements
 * @param exponent  Power (must be >= 2)
 * @return          Max absolute value (0 if all zeros)
 */
double normalize_and_pow_dd(const DDComplex* src, DDComplex* dst,
                            size_t n, int exponent);

/**
 * NUMA-aware first-touch zero for DD arrays.
 * Zeros all elements in parallel so pages are allocated on local NUMA nodes.
 */
void numa_first_touch_zero_dd(DDComplex* data, size_t n);
