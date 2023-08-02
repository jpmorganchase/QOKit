/***************************************
* // SPDX-License-Identifier: Apache-2.0
* // Copyright : JP Morgan Chase & Co
****************************************/ 

#pragma once
#include <stddef.h>

void apply_qaoa_furx(double* sv_real, double* sv_imag, double* const gammas, double* const betas, double* const hc_diag, unsigned int n_qubits, size_t n_states, size_t n_layers);

void apply_qaoa_furxy_ring(double* sv_real, double* sv_imag, double* const gammas, double* const betas, double* const hc_diag, unsigned int n_qubits, size_t n_states, size_t n_layers, size_t n_trotters);

void apply_qaoa_furxy_complete(double* sv_real, double* sv_imag, double* const gammas, double* const betas, double* const hc_diag, unsigned int n_qubits, size_t n_states, size_t n_layers, size_t n_trotters);
