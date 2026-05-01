#pragma once
/**
 * Double-double arithmetic for ~32 significant decimal digits.
 *
 * Compiled with -ffp-contract=off -mno-fma to ensure exact IEEE 754 rounding
 * required by Dekker splitting and Knuth 2Sum.
 *
 * References:
 *   Hida, Li, Bailey (2001). "Library for Double-Double and Quad-Double Arithmetic."
 *   Dekker (1971). "A floating-point technique for extending the available precision."
 */

#include <complex>
#include <cmath>

using C128 = std::complex<double>;

// ── DD real primitives ──────────────────────────────────────────

struct DD {
    double hi, lo;

    DD() : hi(0.0), lo(0.0) {}
    DD(double h) : hi(h), lo(0.0) {}
    DD(double h, double l) : hi(h), lo(l) {}
};

/// Error-free addition (Knuth 2Sum). Returns (s, e) where s + e = a + b exactly.
inline void two_sum(double a, double b, double& s, double& e) {
    s = a + b;
    double v = s - a;
    e = (a - (s - v)) + (b - v);
}

/// Error-free multiplication (Dekker splitting). Returns (p, e) where p + e = a * b exactly.
inline void two_prod(double a, double b, double& p, double& e) {
    p = a * b;
    constexpr double SPLIT = 134217729.0;  // 2^27 + 1
    double t = SPLIT * a;
    double a_hi = t - (t - a), a_lo = a - a_hi;
    t = SPLIT * b;
    double b_hi = t - (t - b), b_lo = b - b_hi;
    e = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
}

/// DD addition
inline DD dd_add(DD a, DD b) {
    double s, e;
    two_sum(a.hi, b.hi, s, e);
    e += a.lo + b.lo;
    double hi, lo;
    two_sum(s, e, hi, lo);
    return {hi, lo};
}

/// DD subtraction
inline DD dd_sub(DD a, DD b) {
    return dd_add(a, {-b.hi, -b.lo});
}

/// DD multiplication
inline DD dd_mul(DD a, DD b) {
    double p, e;
    two_prod(a.hi, b.hi, p, e);
    e += a.hi * b.lo + a.lo * b.hi;
    double hi, lo;
    two_sum(p, e, hi, lo);
    return {hi, lo};
}

/// DD division
inline DD dd_div(DD a, DD b) {
    double q_hi = a.hi / b.hi;
    // Residual: r = a - q_hi * b
    DD qb = dd_mul({q_hi, 0.0}, b);
    DD r = dd_sub(a, qb);
    double q_lo = r.hi / b.hi;
    double hi, lo;
    two_sum(q_hi, q_lo, hi, lo);
    return {hi, lo};
}

/// DD integer power via binary exponentiation
DD dd_pow(DD base, int n);

// Macro for inlined power specializations (n=2..7). Used by dd_pow,
// dd_cpow, and dd_cpow_inline to avoid tripling the same if-chain.
// MUL must be a binary function/macro: MUL(a, b) -> product.
#define DD_POW_SPECIALIZATIONS(base, MUL) \
    if (n == 2) return MUL(base, base); \
    if (n == 3) { auto _b2 = MUL(base, base); return MUL(_b2, base); } \
    if (n == 4) { auto _b2 = MUL(base, base); return MUL(_b2, _b2); } \
    if (n == 5) { auto _b2 = MUL(base, base); auto _b4 = MUL(_b2, _b2); return MUL(_b4, base); } \
    if (n == 6) { auto _b2 = MUL(base, base); auto _b3 = MUL(_b2, base); return MUL(_b3, _b3); } \
    if (n == 7) { auto _b2 = MUL(base, base); auto _b4 = MUL(_b2, _b2); return MUL(_b4, MUL(_b2, base)); }

// ── DD complex ──────────────────────────────────────────────────

struct DDComplex {
    C128 hi, lo;

    DDComplex() : hi(0.0), lo(0.0) {}
    DDComplex(C128 h) : hi(h), lo(0.0) {}
    DDComplex(C128 h, C128 l) : hi(h), lo(l) {}
    DDComplex(double r, double i = 0.0) : hi(C128(r, i)), lo(0.0) {}

    /// Convert to standard complex (loses low-order bits)
    C128 to_c128() const { return hi + lo; }
};

/// Complex DD multiply using Dekker splitting on real components
DDComplex dd_cmul(DDComplex a, DDComplex b);

/// Complex DD add
inline DDComplex dd_cadd(DDComplex a, DDComplex b) {
    // Component-wise two_sum on real and imaginary parts
    double sr, er, si, ei;
    two_sum(a.hi.real(), b.hi.real(), sr, er);
    two_sum(a.hi.imag(), b.hi.imag(), si, ei);
    double lor = a.lo.real() + b.lo.real() + er;
    double loi = a.lo.imag() + b.lo.imag() + ei;
    double hr, lr, hh, lh;
    two_sum(sr, lor, hr, lr);
    two_sum(si, loi, hh, lh);
    return {C128(hr, hh), C128(lr, lh)};
}

/// Complex DD subtract
inline DDComplex dd_csub(DDComplex a, DDComplex b) {
    return dd_cadd(a, {-b.hi, -b.lo});
}

/// Complex DD division
DDComplex dd_cdiv(DDComplex a, DDComplex b);

/// Complex DD integer power
DDComplex dd_cpow(DDComplex base, int n);

/// Precise complex power: promote to DD, compute, truncate back to C128.
/// Safe to call from float64 code — all DD arithmetic stays in dd.cpp
/// (compiled with -ffp-contract=off).
C128 pow_precise(C128 base, int n);


// ── DDComplex operators ──────────────────────────────────────
// Enable generic code shared between C128 and DDComplex paths.
// operator* calls dd_cmul (cross-TU, compiled with -ffp-contract=off).
// operator+/- call dd_cadd/dd_csub (inline, addition-only, FMA-safe).

inline DDComplex operator+(DDComplex a, DDComplex b) { return dd_cadd(a, b); }
inline DDComplex operator-(DDComplex a, DDComplex b) { return dd_csub(a, b); }
inline DDComplex operator*(DDComplex a, DDComplex b) { return dd_cmul(a, b); }
inline DDComplex operator-(DDComplex a) { return {-a.hi, -a.lo}; }

inline DDComplex& operator+=(DDComplex& a, DDComplex b) { a = dd_cadd(a, b); return a; }
inline DDComplex& operator-=(DDComplex& a, DDComplex b) { a = dd_csub(a, b); return a; }
inline DDComplex& operator*=(DDComplex& a, DDComplex b) { a = dd_cmul(a, b); return a; }

inline DDComplex operator*(DDComplex a, double s) { return dd_cmul(a, DDComplex(s)); }
inline DDComplex operator*(double s, DDComplex a) { return dd_cmul(DDComplex(s), a); }


// ── Overloaded helpers for templated code ─────────────────────
// These let template<typename Scalar> code call the right function
// for both C128 and DDComplex without specialization.

inline C128      pow_scalar(C128 x, int n)      { return pow_precise(x, n); }
inline DDComplex pow_scalar(DDComplex x, int n)  { return dd_cpow(x, n); }

inline double abs_scalar(C128 x)      { return std::abs(x); }
inline double abs_scalar(DDComplex x)  { return std::abs(x.to_c128()); }

inline C128      conj_scalar(C128 x)      { return std::conj(x); }
inline DDComplex conj_scalar(DDComplex x) { return {std::conj(x.hi), std::conj(x.lo)}; }

inline double to_real(C128 x)      { return x.real(); }
inline double to_real(DDComplex x) { return x.to_c128().real(); }

inline C128 to_c128(C128 x)      { return x; }
inline C128 to_c128(DDComplex x)  { return x.to_c128(); }
