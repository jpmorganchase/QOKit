/**
 * Elementary doubled tensors and WHT charge contraction implementation.
 */

#include "primitives.h"
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

const double CHARGE_DIAG[4][4] = {
    { 1,  1,  1,  1},  // a=0: identity
    { 1, -1,  1, -1},  // a=1: Z_bra
    { 1,  1, -1, -1},  // a=2: Z_ket
    { 1, -1, -1,  1},  // a=3: Z_ket * Z_bra
};


void doubled_mixer(double beta, C128* M) {
    double c = std::cos(beta);
    double s = std::sin(beta);
    // Rx = [[c, -is], [-is, c]]
    // M = kron(Rx, conj(Rx))
    C128 Rx[2][2] = {
        {C128(c, 0), C128(0, -s)},
        {C128(0, -s), C128(c, 0)}
    };
    C128 RxC[2][2] = {
        {C128(c, 0), C128(0, s)},
        {C128(0, s), C128(c, 0)}
    };
    // kron: M[i*2+j, k*2+l] = Rx[i,k] * RxC[j,l]
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                for (int l = 0; l < 2; l++)
                    M[(i * 2 + j) * 4 + (k * 2 + l)] = Rx[i][k] * RxC[j][l];
}


void charge_weight_matrix(double gamma, C128* W) {
    double c = std::cos(gamma);
    double s = std::sin(gamma);
    double c2 = c * c, s2 = s * s;
    C128 ics(0.0, c * s);

    // zk = [1, 1, -1, -1], zb = [1, -1, 1, -1]
    double zk[4] = {1, 1, -1, -1};
    double zb[4] = {1, -1, 1, -1};

    for (int h = 0; h < 4; h++) {
        W[h * 4 + 0] = C128(c2, 0);           // a=0: cos^2
        W[h * 4 + 1] = ics * zb[h];           // a=1: i*c*s*zb
        W[h * 4 + 2] = -ics * zk[h];          // a=2: -i*c*s*zk
        W[h * 4 + 3] = C128(s2 * zk[h] * zb[h], 0);  // a=3: sin^2*zk*zb
    }
}


void root_charge_weights(double gamma, C128* u) {
    double c = std::cos(gamma);
    double s = std::sin(gamma);
    u[0] = C128(c * c, 0);
    u[1] = C128(0, c * s);
    u[2] = C128(0, -c * s);
    u[3] = C128(s * s, 0);
}


void doubled_mixer_deriv(double beta, C128* dM) {
    double c = std::cos(beta);
    double s = std::sin(beta);
    C128 Rx[2][2] = {
        {C128(c, 0), C128(0, -s)},
        {C128(0, -s), C128(c, 0)}
    };
    C128 dRx[2][2] = {
        {C128(-s, 0), C128(0, -c)},
        {C128(0, -c), C128(-s, 0)}
    };
    C128 RxC[2][2] = {
        {C128(c, 0), C128(0, s)},
        {C128(0, s), C128(c, 0)}
    };
    C128 dRxC[2][2] = {
        {C128(-s, 0), C128(0, c)},
        {C128(0, c), C128(-s, 0)}
    };
    // dM/dbeta = kron(dRx, conj(Rx)) + kron(Rx, conj(dRx))
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                for (int l = 0; l < 2; l++)
                    dM[(i * 2 + j) * 4 + (k * 2 + l)] =
                        dRx[i][k] * RxC[j][l] + Rx[i][k] * dRxC[j][l];
}


void charge_weight_matrix_deriv(double gamma, C128* dW) {
    double c = std::cos(gamma);
    double s = std::sin(gamma);
    double c2 = c * c, s2 = s * s;
    double dc2 = -2 * c * s;
    double ds2 = 2 * c * s;
    C128 dics(0.0, c2 - s2);

    double zk[4] = {1, 1, -1, -1};
    double zb[4] = {1, -1, 1, -1};

    for (int h = 0; h < 4; h++) {
        dW[h * 4 + 0] = C128(dc2, 0);
        dW[h * 4 + 1] = dics * zb[h];
        dW[h * 4 + 2] = -dics * zk[h];
        dW[h * 4 + 3] = C128(ds2 * zk[h] * zb[h], 0);
    }
}


void root_charge_weights_deriv(double gamma, C128* du) {
    double c = std::cos(gamma);
    double s = std::sin(gamma);
    double c2 = c * c, s2 = s * s;
    du[0] = C128(-2 * c * s, 0);
    du[1] = C128(0, c2 - s2);
    du[2] = C128(0, -(c2 - s2));
    du[3] = C128(2 * c * s, 0);
}


void wht_charge_contract(const C128* M, const C128* T, C128* out,
                          size_t R, size_t rest) {
    // T layout: [i, sigma, b, r] = T[i*16*rest + sigma*4*rest + b*rest + r]
    // out layout: [a, i, b, r] = out[a*R*4*rest + i*4*rest + b*rest + r]
    // where a = charge channel (0..3)
    size_t vec_len = 4 * rest;  // one row of output = 4 b-values * rest

    // collapse(3) exposes R * 4 * rest iterations to all cores.
    // At p=10: R=4^8=65536, rest=1 → 262144 iterations.
    // At p=6 level 3: R=16, rest=64 → 4096 iterations.
    // Both cases saturate 91 cores well.
    #pragma omp parallel for schedule(static) collapse(3)
    for (size_t i = 0; i < R; i++) {
        for (size_t b = 0; b < 4; b++) {
            for (size_t r = 0; r < rest; r++) {
                // Precompute base offsets to reduce address arithmetic
                size_t t_base = i * 16 * rest + b * rest + r;
                size_t stride = 4 * rest;  // distance between sigma values

                // Load e[sigma] = M[b, sigma] * T[i, sigma, b, r]
                C128 e0 = M[b * 4 + 0] * T[t_base + 0 * stride];
                C128 e1 = M[b * 4 + 1] * T[t_base + 1 * stride];
                C128 e2 = M[b * 4 + 2] * T[t_base + 2 * stride];
                C128 e3 = M[b * 4 + 3] * T[t_base + 3 * stride];

                // WHT butterfly: 4 muls + 8 adds (vs 16 muls + 12 adds naive)
                C128 p02 = e0 + e2, q02 = e0 - e2;
                C128 p13 = e1 + e3, q13 = e1 - e3;

                size_t base_out = i * vec_len + b * rest + r;
                out[0 * R * vec_len + base_out] = p02 + p13;  // a=0
                out[1 * R * vec_len + base_out] = p02 - p13;  // a=1
                out[2 * R * vec_len + base_out] = q02 + q13;  // a=2
                out[3 * R * vec_len + base_out] = q02 - q13;  // a=3
            }
        }
    }
}


void wht_charge_contract_dd(const DDComplex* M, const DDComplex* T,
                             DDComplex* out, size_t R, size_t rest) {
    size_t vec_len = 4 * rest;

    #pragma omp parallel for schedule(static) collapse(3)
    for (size_t i = 0; i < R; i++) {
        for (size_t b = 0; b < 4; b++) {
            for (size_t r = 0; r < rest; r++) {
                size_t t_base = i * 16 * rest + b * rest + r;
                size_t stride = 4 * rest;
                DDComplex e0 = dd_cmul(M[b * 4 + 0], T[t_base + 0 * stride]);
                DDComplex e1 = dd_cmul(M[b * 4 + 1], T[t_base + 1 * stride]);
                DDComplex e2 = dd_cmul(M[b * 4 + 2], T[t_base + 2 * stride]);
                DDComplex e3 = dd_cmul(M[b * 4 + 3], T[t_base + 3 * stride]);

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
