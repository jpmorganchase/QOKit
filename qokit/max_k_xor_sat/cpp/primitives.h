#pragma once
/**
 * Elementary doubled tensors and WHT charge contraction.
 */

#include <complex>
#include <array>
#include <vector>
#include "dd.h"

using C128 = std::complex<double>;

/// Charge diagonal: CDIAG[a][sigma]
extern const double CHARGE_DIAG[4][4];

/// 4x4 doubled mixer M = kron(Rx(beta), conj(Rx(beta)))
/// Output is stored in M[16] in row-major order (M[row*4 + col]).
void doubled_mixer(double beta, C128* M);

/// 4x4 charge weight matrix W[h, a] = w_a(h, gamma)
/// Output is stored in W[16] in row-major order (W[h*4 + a]).
void charge_weight_matrix(double gamma, C128* W);

/// Root charge weights u[4] = [cos^2, i*c*s, -i*c*s, sin^2]
void root_charge_weights(double gamma, C128* u);

/// dM/dbeta for doubled mixer
void doubled_mixer_deriv(double beta, C128* dM);

/// dW/dgamma for charge weight matrix
void charge_weight_matrix_deriv(double gamma, C128* dW);

/// du/dgamma for root charge weights
void root_charge_weights_deriv(double gamma, C128* du);

/**
 * WHT butterfly charge contraction for all 4 channels simultaneously.
 *
 * Computes out[a][i, b, r] = sum_sigma CDIAG[a, sigma] * M[b, sigma] * T[i, sigma, b, r]
 * for all 4 charge channels a=0..3.
 *
 * T layout (row-major): T[i * 16*rest + sigma * 4*rest + b * rest + r]
 * out layout: out[a * R*4*rest + i * 4*rest + b * rest + r]
 *   where a is charge channel, i is batch index (R rows), b is sigma_out, r is rest.
 *
 * @param M     4x4 base mixer (row-major, M[b*4 + sigma])
 * @param T     input tensor, shape (R, 4, 4, rest) flat
 * @param out   output tensor, shape (4*R, 4, rest) flat — 4 charge channels concatenated
 * @param R     number of input rows
 * @param rest  trailing dimension size
 */
void wht_charge_contract(const C128* M, const C128* T, C128* out,
                          size_t R, size_t rest);

/// DD version of WHT charge contraction
void wht_charge_contract_dd(const DDComplex* M, const DDComplex* T,
                             DDComplex* out, size_t R, size_t rest);
