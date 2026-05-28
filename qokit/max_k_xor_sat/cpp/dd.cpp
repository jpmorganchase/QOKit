/**
 * Double-double arithmetic implementation.
 *
 * MUST be compiled with -ffp-contract=off -mno-fma to ensure exact IEEE 754
 * rounding for Dekker splitting and Knuth 2Sum correctness.
 */

#include "dd.h"
#include <limits>

static_assert(std::numeric_limits<double>::is_iec559,
              "DD arithmetic requires IEEE 754 double-precision floats");

DD dd_pow(DD base, int n) {
    if (n < 0) return {0.0, 0.0};
    if (n == 0) return {1.0, 0.0};
    if (n == 1) return base;
    DD_POW_SPECIALIZATIONS(base, dd_mul);

    DD result = {1.0, 0.0};
    DD b = base;
    int exp = n;
    while (exp > 0) {
        if (exp & 1) result = dd_mul(result, b);
        b = dd_mul(b, b);
        exp >>= 1;
    }
    return result;
}


DDComplex dd_cmul(DDComplex a, DDComplex b) {
    double ar = a.hi.real(), ai = a.hi.imag();
    double br = b.hi.real(), bi = b.hi.imag();

    double p1, e1, p2, e2;
    two_prod(ar, br, p1, e1);
    two_prod(ai, bi, p2, e2);
    double real_hi = p1 - p2;
    double real_lo = e1 - e2 + ((p1 - real_hi) - p2);

    double p3, e3, p4, e4;
    two_prod(ar, bi, p3, e3);
    two_prod(ai, br, p4, e4);
    double imag_hi = p3 + p4;
    double imag_lo = e3 + e4 + ((p3 - imag_hi) + p4);

    C128 hi(real_hi, imag_hi);
    C128 lo(real_lo, imag_lo);
    lo += a.hi * b.lo + a.lo * b.hi;

    double sr, er, sii, eii;
    two_sum(hi.real(), lo.real(), sr, er);
    two_sum(hi.imag(), lo.imag(), sii, eii);
    return {C128(sr, sii), C128(er, eii)};
}


DDComplex dd_cdiv(DDComplex a, DDComplex b) {
    C128 q_hi = a.hi / b.hi;
    DDComplex qb = dd_cmul({q_hi}, b);
    DDComplex r = dd_csub(a, qb);
    C128 q_lo = r.hi / b.hi;
    double sr, er, si, ei;
    two_sum(q_hi.real(), q_lo.real(), sr, er);
    two_sum(q_hi.imag(), q_lo.imag(), si, ei);
    return {C128(sr, si), C128(er, ei)};
}


DDComplex dd_cpow(DDComplex base, int n) {
    if (n < 0) return {C128(0.0), C128(0.0)};
    if (n == 0) return {C128(1.0), C128(0.0)};
    if (n == 1) return base;
    DD_POW_SPECIALIZATIONS(base, dd_cmul);

    DDComplex result = {C128(1.0), C128(0.0)};
    DDComplex b = base;
    int exp = n;
    while (exp > 0) {
        if (exp & 1) result = dd_cmul(result, b);
        b = dd_cmul(b, b);
        exp >>= 1;
    }
    return result;
}


C128 pow_precise(C128 base, int n) {
    return dd_cpow(DDComplex(base), n).to_c128();
}
